"""Token pruning utilities for ViT verification.

Provides a ``MaskedSoftmax`` custom ONNX op with efficient zonotope and
differential handlers that follow the DeepT decomposition internally
(pairwise diff → exp linearizer on concrete bounds → sum → reciprocal).
No bilinear elementwise products — same memory profile as standard DeepT.

Usage::

    from pruning import (
        MaskedPostConcat, build_emb_mask, build_full_emb_mask,
        classify_topk, enumerate_pruning_cases,
        certify_pruned_sample_diff,
    )
"""
from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab import utils
from boundlab.expr._affine import AffineSum, ConstVal
from boundlab.expr._core import Expr
from boundlab.expr._var import LpEpsilon
from boundlab.interp import _onnx_broadcast
from boundlab.interp.onnx import onnx_export
from boundlab.linearop import PadOp
from boundlab.zono.bilinear import bilinear_elementwise
from boundlab.zono.exp import exp_linearizer
from boundlab.zono.reciprocal import reciprocal_linearizer


# ---------------------------------------------------------------------------
# Register bilinear elementwise Mul (needed for attn_w @ V in diff path)
# ---------------------------------------------------------------------------

def _mul_handler(X, Y):
    X, Y = _onnx_broadcast(X, Y)
    if isinstance(X, Expr) and isinstance(Y, Expr):
        return bilinear_elementwise(X, Y)
    return X * Y

zono.interpret["Mul"] = _mul_handler


# ---------------------------------------------------------------------------
# Custom ONNX op: MaskedSoftmax
# ---------------------------------------------------------------------------

def masked_softmax_op(scores: Tensor, col_mask: Tensor) -> Tensor:
    """Custom ONNX op for masked softmax.

    At concrete runtime returns zeros (use ``use_custom_op=False`` for MC).
    During ONNX export creates a ``boundlab::MaskedSoftmax`` node that the
    zonotope and differential interpreters handle efficiently.

    Args:
        scores: Raw attention scores, shape ``(h, n, n)``.
        col_mask: Column mask, shape ``(n,)``.  1 for kept, 0 for pruned.
    """
    return torch.onnx.ops.symbolic(
        "boundlab::MaskedSoftmax",
        (scores, col_mask),
        dtype=scores.dtype,
        shape=scores.shape,
        version=1,
    )


# ---------------------------------------------------------------------------
# Zonotope handler: follows softmax_handler with mask on concrete coefficients
# ---------------------------------------------------------------------------

def masked_softmax_zono_handler(x, col_mask) -> Expr:
    """Efficient masked softmax for zonotope abstract interpretation.

    Follows the standard ``softmax_handler`` pattern: pairwise diff,
    exp linearizer on concrete bounds, sum, reciprocal.  The col_mask
    is multiplied into the concrete coefficients before summing — no
    bilinear Expr × Expr products.

    DeepT decomposition:
        masked_softmax(s)_k = col_mask[k] / Σ_j col_mask[j] * exp(s_j - s_k)
    """
    if isinstance(x, torch.Tensor):
        x = ConstVal(x)
    if not isinstance(x, Expr):
        return NotImplemented

    if isinstance(col_mask, Expr):
        # col_mask should be a constant (from initializer)
        col_mask = prop.center(col_mask)

    # Pairwise diff: diff[..., i, j] = x_j - x_i  (last dim)
    dim = len(x.shape) - 1
    diff = -utils.pairwise_diff(x, dim)     # Expr, shape (..., n, n)
    ub, lb = diff.ublb()                     # concrete bounds

    # Exp linearizer on concrete bounds → (weights, bias, error)
    expbounds = exp_linearizer(ub, lb)
    weights = expbounds.input_weights[0]     # concrete tensor
    bias = expbounds.bias
    error = expbounds.error_coeffs.tensor

    # Finite-value safety (same as standard handler)
    finite_mask = (torch.isfinite(weights) & torch.isfinite(error)
                   & torch.isfinite(bias) & (lb < 30) & (ub < 30))
    weights = torch.where(finite_mask, weights, 0)
    bias = torch.where(finite_mask, bias, 0)
    error = torch.where(finite_mask, error, 0)

    # --- MASK: multiply col_mask into concrete coefficients before sum ---
    # col_mask shape (n,) broadcasts to last dim (j = sum index)
    weights = weights * col_mask
    bias = bias * col_mask
    error = error * col_mask

    # Sum over j (last dim) → shape (..., n)
    sum_exp = ((weights * diff).sum(dim=-1)
               + error.sum(dim=-1) * LpEpsilon(diff.shape[:-2], reason="masked_softmax_exp")
               + bias.sum(dim=-1))
    finite_mask = finite_mask.all(dim=-1)

    # Tighten sum bounds (same as standard handler, but with mask)
    sum_exp_ub, sum_exp_lb = sum_exp.ublb()
    sum_exp_ub = torch.minimum(sum_exp_ub, (torch.exp(ub) * col_mask).sum(dim=-1))
    sum_exp_lb = torch.maximum(sum_exp_lb, (torch.exp(lb) * col_mask).sum(dim=-1))

    # Reciprocal linearizer
    bounds = reciprocal_linearizer(sum_exp_ub, sum_exp_lb)
    w = bounds.input_weights[0]
    mu = bounds.bias
    beta = bounds.error_coeffs.tensor

    result = finite_mask * (w * sum_exp + mu
                            + beta * LpEpsilon(sum_exp.shape, reason="masked_softmax_recip"))

    # --- MASK: zero pruned key outputs ---
    # result shape: (h, n, n) — col_mask must match exactly for Expr multiply
    target_shape = list(result.shape)
    col_mask_nd = col_mask.view(*([1] * (len(target_shape) - 1)), -1).expand(target_shape).contiguous()
    result = result * col_mask_nd

    return result


# Register for standard zonotope interpreter
zono.interpret["MaskedSoftmax"] = masked_softmax_zono_handler


# ---------------------------------------------------------------------------
# Differential handler: follows diff_softmax_handler with mask insertion
# ---------------------------------------------------------------------------

def diff_masked_softmax_handler(x, col_mask):
    """Differential masked softmax handler.

    Follows ``diff_softmax_handler`` pattern: pairwise diff → diff exp →
    mask (const multiply, preserves correlations) → sum → diff reciprocal.
    Same memory as standard diff softmax — the mask only adds linear ops.

    Falls back to ``masked_softmax_zono_handler`` for plain Expr input.
    """
    from boundlab.diff.expr import DiffExpr2, DiffExpr3
    from boundlab.diff.zono3 import interpret as _diff_interpret

    # Extract concrete col_mask for each network
    if isinstance(col_mask, (DiffExpr2, DiffExpr3)):
        col_mask_x = prop.center(col_mask.x) if isinstance(col_mask.x, Expr) else col_mask.x
        col_mask_y = prop.center(col_mask.y) if isinstance(col_mask.y, Expr) else col_mask.y
    elif isinstance(col_mask, torch.Tensor):
        col_mask_x = col_mask_y = col_mask
    elif isinstance(col_mask, Expr):
        col_mask_x = col_mask_y = prop.center(col_mask)
    else:
        return NotImplemented

    delta_mask = col_mask_x - col_mask_y  # 1 on pruned-only, 0 elsewhere

    # Plain Expr → efficient standard handler
    if isinstance(x, Expr) and not isinstance(x, (DiffExpr2, DiffExpr3)):
        return masked_softmax_zono_handler(x, col_mask_x)

    if isinstance(x, DiffExpr2):
        x = DiffExpr3(x.x, x.y, x.x - x.y)
    if not isinstance(x, DiffExpr3):
        return NotImplemented

    exp_handler = _diff_interpret["Exp"]
    reciprocal_handler = _diff_interpret["Reciprocal"]

    dim = len(x.shape) - 1
    n = x.shape[dim]

    # --- 1. Pairwise diff (same as diff_softmax_handler) ---
    x_i = x.unsqueeze(dim + 1)
    x_j = x.unsqueeze(dim)
    broadcast_shape = list(x.shape)
    broadcast_shape.insert(dim + 1, n)
    x_i_exp = x_i.expand(*broadcast_shape)
    x_j_exp = x_j.expand(*broadcast_shape)
    x_shifted = x_j_exp - x_i_exp  # DiffExpr3, shape (..., n, n)

    # --- 2. Exp (diff handler preserves correlations) ---
    exp_shifted = exp_handler(x_shifted)  # DiffExpr3

    # --- 3. Mask on j-axis (const multiply — preserves correlations) ---
    # j is at dim+1 in the expanded shape
    j_axis = dim + 1
    # Expand masks to match exp_shifted shape
    mask_shape = [1] * len(exp_shifted.shape)
    mask_shape[j_axis] = n
    col_mask_x_j = col_mask_x.view(mask_shape).expand(exp_shifted.shape).contiguous()
    col_mask_y_j = col_mask_y.view(mask_shape).expand(exp_shifted.shape).contiguous()
    delta_mask_j = col_mask_x_j - col_mask_y_j

    # Multiply: a*c - b*d = a*(c-d) + (a-b)*d → preserves exp diff correlation
    exp_masked = DiffExpr3(
        exp_shifted.x * col_mask_x_j,
        exp_shifted.y * col_mask_y_j,
        exp_shifted.x * delta_mask_j + exp_shifted.diff * col_mask_y_j,
    )

    # --- 4. Sum over j ---
    sum_exp = DiffExpr3(
        exp_masked.x.sum(dim=j_axis, keepdim=False),
        exp_masked.y.sum(dim=j_axis, keepdim=False),
        exp_masked.diff.sum(dim=j_axis, keepdim=False),
    )

    # --- 5. Reciprocal (diff handler preserves correlations) ---
    result = reciprocal_handler(sum_exp)  # DiffExpr3

    # --- 6. Mask on output (k-axis = last dim) ---
    k_shape = [1] * len(result.shape)
    k_shape[-1] = n
    col_mask_x_k = col_mask_x.view(k_shape).expand(result.shape).contiguous()
    col_mask_y_k = col_mask_y.view(k_shape).expand(result.shape).contiguous()
    delta_mask_k = col_mask_x_k - col_mask_y_k

    result_final = DiffExpr3(
        result.x * col_mask_x_k,
        result.y * col_mask_y_k,
        result.x * delta_mask_k + result.diff * col_mask_y_k,
    )

    return result_final


# Register for differential interpreter
from boundlab.diff.zono3 import interpret as diff_interpret
diff_interpret["MaskedSoftmax"] = diff_masked_softmax_handler


# ---------------------------------------------------------------------------
# Pipeline sub-models
# ---------------------------------------------------------------------------

class PatchifyStage(nn.Module):
    """``(C, H, W) -> (num_patches, dim)``.  Pure linear, no CLS token."""

    def __init__(self, vit, normalize: bool = False,
                 mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.patch_embed = vit.to_patch_embedding
        self.normalize = normalize
        if normalize:
            self.register_buffer("mean", torch.tensor(float(mean)))
            self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, img: Tensor) -> Tensor:
        if self.normalize:
            img = (img - self.mean) / self.std
        return self.patch_embed(img)


class ScoringModel(nn.Module):
    """``(N+1, D) embeddings -> (N,) importance`` per patch token.

    Parameters
    ----------
    vit : ViT
        Source model.
    score_layer : int
        Which layer's CLS attention to use.  Propagates through layers
        ``0..score_layer-1`` (unmasked) first.
    """

    def __init__(self, vit, score_layer: int = 0):
        super().__init__()
        assert 0 <= score_layer < len(vit.transformer.layers)
        self.score_layer = score_layer
        self.prefix_layers = nn.ModuleList()
        for i in range(score_layer):
            attn_block, ff_block = vit.transformer.layers[i]
            self.prefix_layers.append(nn.ModuleList([attn_block, ff_block]))
        attn_block = vit.transformer.layers[score_layer][0]
        self.norm = attn_block.fn.norm
        self.attn = attn_block.fn.fn
        self.heads = self.attn.heads
        self.dim_head = self.attn.dim_head
        self.scale = self.attn.scale

    def forward(self, x: Tensor) -> Tensor:
        for attn_block, ff_block in self.prefix_layers:
            x = attn_block(x)
            x = ff_block(x)
        xn = self.norm(x)
        n = xn.shape[0]
        h, d = self.heads, self.dim_head
        Q = self.attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)
        K = self.attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
        Q_cls = Q[:, 0:1, :]
        scores = (Q_cls @ K.transpose(-2, -1)) * self.scale
        attn_weights = scores.softmax(dim=-1)
        importance = attn_weights.mean(dim=0).squeeze(0)
        return importance[1:]


class MaskedPostConcat(nn.Module):
    """Transformer + head with pruning via masked softmax.

    Parameters
    ----------
    vit : ViT
        Source model (any depth).
    emb_mask : Tensor
        Shape ``(N+1, D)``.  Ones for kept, zeros for pruned.
    mask_from_layer : int
        First layer at which pruning applies.
    use_custom_op : bool
        If True, uses the ``MaskedSoftmax`` custom ONNX op (for
        verification — efficient handler, returns zeros at runtime).
        If False, uses concrete pairwise-diff computation (for MC).
    """

    def __init__(self, vit, emb_mask: Tensor, mask_from_layer: int = 0,
                 use_custom_op: bool = False):
        super().__init__()
        self.pool = vit.pool
        self.mlp_head = vit.mlp_head
        self.mask_from_layer = mask_from_layer
        self.use_custom_op = use_custom_op
        self.register_buffer("emb_mask", emb_mask)

        self.n_layers = len(vit.transformer.layers)
        self.attn_norms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.ff_blocks = nn.ModuleList()
        self.heads_list = []
        self.dim_head_list = []
        self.scale_list = []
        for attn_block, ff_block in vit.transformer.layers:
            self.attn_norms.append(attn_block.fn.norm)
            self.attns.append(attn_block.fn.fn)
            self.ff_blocks.append(ff_block)
            self.heads_list.append(attn_block.fn.fn.heads)
            self.dim_head_list.append(attn_block.fn.fn.dim_head)
            self.scale_list.append(attn_block.fn.fn.scale)

    def forward(self, x: Tensor) -> Tensor:
        col_mask = self.emb_mask[:, 0]  # (N+1,)

        for layer_idx in range(self.n_layers):
            if layer_idx == self.mask_from_layer:
                x = x * self.emb_mask

            n = x.shape[0]
            h = self.heads_list[layer_idx]
            d = self.dim_head_list[layer_idx]
            scale = self.scale_list[layer_idx]
            attn = self.attns[layer_idx]

            residual = x
            xn = self.attn_norms[layer_idx](x)
            q = attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)
            k = attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
            v = attn.to_v(xn).reshape(n, h, d).permute(1, 0, 2)
            raw_scores = (q @ k.transpose(-2, -1)) * scale

            if layer_idx >= self.mask_from_layer:
                if self.use_custom_op:
                    # Custom ONNX op → efficient handler during verification
                    attn_w = masked_softmax_op(raw_scores, col_mask)
                else:
                    # Concrete computation for MC
                    diff = raw_scores.unsqueeze(-2) - raw_scores.unsqueeze(-1)
                    exp_diff = torch.exp(diff)
                    col_mask_j = col_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
                    exp_masked = exp_diff * col_mask_j
                    sum_exp = exp_masked.sum(dim=-1)
                    col_mask_k = col_mask.unsqueeze(0).unsqueeze(0)
                    attn_w = torch.reciprocal(sum_exp) * col_mask_k
            else:
                attn_w = raw_scores.softmax(dim=-1)

            out = (attn_w @ v).permute(1, 0, 2).reshape(n, h * d)
            out = attn.to_out(out)
            x = residual + out
            x = self.ff_blocks[layer_idx](x)

        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_emb_mask(num_tokens: int, dim: int, kept_patches: set[int]) -> Tensor:
    """Build embedding mask ``(N+1, D)``.  CLS always kept."""
    total = num_tokens + 1
    emb_mask = torch.zeros(total, dim)
    emb_mask[0] = 1.0
    for p in kept_patches:
        emb_mask[p + 1] = 1.0
    return emb_mask


def build_full_emb_mask(num_tokens: int, dim: int) -> Tensor:
    """All-ones mask (no pruning)."""
    return build_emb_mask(num_tokens, dim, set(range(num_tokens)))


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

def classify_topk(ub_scores, lb_scores, K):
    N = len(ub_scores)
    n_prune = N - K
    wins = torch.zeros(N, dtype=torch.long)
    losses = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if lb_scores[i] > ub_scores[j]:
                wins[i] += 1
            if ub_scores[i] < lb_scores[j]:
                losses[i] += 1
    definite_keep = {i for i in range(N) if wins[i] >= n_prune}
    definite_prune = {i for i in range(N) if losses[i] >= K}
    uncertain = set(range(N)) - definite_keep - definite_prune
    return definite_keep, definite_prune, uncertain


def enumerate_pruning_cases(definite_keep, uncertain, K):
    K_remaining = K - len(definite_keep)
    if K_remaining < 0:
        return [definite_keep.copy()]
    if K_remaining >= len(uncertain):
        return [definite_keep | uncertain]
    if K_remaining == 0:
        return [definite_keep.copy()]
    return [definite_keep | set(c) for c in combinations(sorted(uncertain), K_remaining)]


# ---------------------------------------------------------------------------
# Zonotope construction
# ---------------------------------------------------------------------------

def build_zonotope_no_cat(vit, img, eps, op_patch):
    num_patches = (img.shape[-1] // vit.patch_size) ** 2
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    patch_zono = op_patch(expr.ConstVal(img) + eps * expr.LpEpsilon(list(img.shape)))
    pad_op = PadOp(patch_zono.shape, [0, 0, 1, 0])
    padded = AffineSum((pad_op, patch_zono))
    cls_padded = F.pad(vit.cls_token[0], [0, 0, 0, num_patches])
    return padded + ConstVal(cls_padded + vit.pos_embedding[0])


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_scoring(vit, num_tokens, dim, score_layer=0):
    scoring = ScoringModel(vit, score_layer=score_layer).eval()
    gm = onnx_export(scoring, ([num_tokens + 1, dim],))
    return zono.interpret(gm), scoring


def export_patchify(vit, img_shape, normalize=False, mean=0.0, std=1.0):
    patchify = PatchifyStage(vit, normalize, mean, std).eval()
    gm = onnx_export(patchify, (img_shape,))
    return zono.interpret(gm), patchify


def export_masked_post_concat(vit, emb_mask, num_tokens, dim, mask_from_layer=0):
    """Export with custom op enabled (for verification)."""
    model = MaskedPostConcat(vit, emb_mask, mask_from_layer=mask_from_layer,
                              use_custom_op=True).eval()
    return onnx_export(model, ([num_tokens + 1, dim],))


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------

class CertifyResult(NamedTuple):
    ub: Tensor
    lb: Tensor
    n_cases: int
    definite_keep: set[int]
    definite_prune: set[int]
    uncertain: set[int]


def certify_pruned_sample_diff(
    vit, img, eps, K, op_patch, op_score,
    num_tokens, dim, mask_from_layer=0,
):
    from boundlab.diff.expr import DiffExpr3
    from boundlab.diff.net import diff_net

    full_zono = build_zonotope_no_cat(vit, img, eps, op_patch)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    ub_scores, lb_scores = op_score(full_zono).ublb()
    definite_keep, definite_prune, uncertain = classify_topk(ub_scores, lb_scores, K)
    cases = enumerate_pruning_cases(definite_keep, uncertain, K)

    gm_full = export_masked_post_concat(
        vit, build_full_emb_mask(num_tokens, dim), num_tokens, dim, mask_from_layer,
    )

    best_ub = best_lb = None
    for kept in cases:
        gm_pruned = export_masked_post_concat(
            vit, build_emb_mask(num_tokens, dim, kept), num_tokens, dim, mask_from_layer,
        )
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        merged = diff_net(gm_full, gm_pruned)
        out = diff_interpret(merged)(full_zono)
        if isinstance(out, DiffExpr3):
            d_ub, d_lb = out.diff.ublb()
        else:
            d_ub, d_lb = (out.x - out.y).ublb()
        if best_ub is None:
            best_ub, best_lb = d_ub.clone(), d_lb.clone()
        else:
            best_ub = torch.maximum(best_ub, d_ub)
            best_lb = torch.minimum(best_lb, d_lb)

    return CertifyResult(best_ub, best_lb, len(cases),
                         definite_keep, definite_prune, uncertain)