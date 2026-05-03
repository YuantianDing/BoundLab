"""Token pruning utilities for ViT verification.

Reusable components for certifying top-K token pruning in Vision Transformers.
Handles **both** the embedding mask (zeroing pruned token rows) and the
softmax mask (zeroing pruned columns in the exponentiated attention scores).

Typical usage::

    from pruning import (
        PatchifyStage, ScoringModel, MaskedPostConcat,
        build_zonotope_no_cat, classify_topk, enumerate_pruning_cases,
        build_emb_mask, certify_pruned_sample_diff,
    )

    vit = build_mnist_vit("mnist_transformer.pt")
    ...
    result = certify_pruned_sample_diff(vit, img, eps, K, op_patch, op_score, 16, 64)

Design notes
------------
* ``MaskedPostConcat`` takes a single **embedding mask** ``(N+1, D)`` and
  derives a column mask ``(1, 1, N+1)`` from it inside ``forward()``.

* The **softmax mask** works by multiplying ``exp(scores)`` by the 0/1
  column mask before normalizing::

      exp_scores = exp(Q @ K^T * scale)
      exp_masked = exp_scores * col_mask   # pruned columns -> 0
      attn       = exp_masked / sum(exp_masked, dim=-1)

  This is exact (no ``exp(-50) ≈ 0`` approximation) and diff-friendly:
  ``diff_net`` only pairs 0/1 mask values, and the ``exp × mask`` product
  is a standard bilinear op handled by McCormick relaxation.
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
from boundlab.expr._affine import AffineSum, ConstVal
from boundlab.expr._core import Expr
from boundlab.interp import _onnx_broadcast
from boundlab.interp.onnx import onnx_export
from boundlab.linearop import PadOp
from boundlab.zono.bilinear import bilinear_elementwise


# ---------------------------------------------------------------------------
# Register bilinear elementwise Mul for the standard zonotope interpreter.
# Without this, Mul(Expr, Expr) fails — needed for exp(scores) * col_mask.
# ---------------------------------------------------------------------------

def _mul_handler(X, Y):
    X, Y = _onnx_broadcast(X, Y)
    if isinstance(X, Expr) and isinstance(Y, Expr):
        return bilinear_elementwise(X, Y)
    return X * Y

zono.interpret["Mul"] = _mul_handler


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

    Computes mean-over-heads CLS attention weights at a chosen layer.
    If ``score_layer > 0``, the model first propagates through layers
    ``0..score_layer-1`` (standard attention, no pruning) to obtain
    the embeddings at that layer's input.

    Parameters
    ----------
    vit : ViT
        The source ViT model.
    score_layer : int
        Which transformer layer's CLS attention to use for scoring.
        Default 0 (first layer).
    """

    def __init__(self, vit, score_layer: int = 0):
        super().__init__()
        assert 0 <= score_layer < len(vit.transformer.layers), \
            f"score_layer={score_layer} but model has {len(vit.transformer.layers)} layers"

        self.score_layer = score_layer

        # Store prefix layers (0..score_layer-1) for propagation
        self.prefix_layers = nn.ModuleList()
        for i in range(score_layer):
            attn_block, ff_block = vit.transformer.layers[i]
            self.prefix_layers.append(nn.ModuleList([attn_block, ff_block]))

        # Scoring layer's attention components
        attn_block = vit.transformer.layers[score_layer][0]
        self.norm = attn_block.fn.norm
        self.attn = attn_block.fn.fn
        self.heads = self.attn.heads
        self.dim_head = self.attn.dim_head
        self.scale = self.attn.scale

    def forward(self, x: Tensor) -> Tensor:
        # Propagate through prefix layers (standard attention, no masking)
        for attn_block, ff_block in self.prefix_layers:
            x = attn_block(x)
            x = ff_block(x)

        # Compute CLS attention at the scoring layer
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

    Handles any number of transformer layers.  Takes a single **embedding
    mask** ``(N+1, D)`` and derives the softmax column mask from it.
    At every attention layer, softmax is replaced by::

        exp_scores = exp(Q @ K^T * scale)
        exp_masked = exp_scores * col_mask   # col_mask = emb_mask[:, 0]
        attn_w     = exp_masked * reciprocal(sum(exp_masked))

    This is exact (pruned columns contribute exactly zero to softmax) and
    keeps all paired initializer differences in {0, 1} for ``diff_net``.

    Parameters
    ----------
    vit : ViT
        The source ViT model (any depth).
    emb_mask : Tensor
        Shape ``(N+1, D)``.  Ones for kept tokens, zeros for pruned.
        CLS (row 0) must always be 1.
    """

    def __init__(self, vit, emb_mask: Tensor):
        super().__init__()
        self.pool = vit.pool
        self.mlp_head = vit.mlp_head
        self.register_buffer("emb_mask", emb_mask)  # (N+1, D)

        # Store per-layer attention + FF components
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
        # --- Embedding mask ---
        x = x * self.emb_mask

        # --- Derive softmax column mask ---
        col_mask = self.emb_mask[:, 0]                        # (N+1,) 1/0
        col_mask = col_mask.unsqueeze(0).unsqueeze(0)         # (1, 1, N+1)

        # --- Transformer layers ---
        for layer_idx in range(self.n_layers):
            n = x.shape[0]
            h = self.heads_list[layer_idx]
            d = self.dim_head_list[layer_idx]
            scale = self.scale_list[layer_idx]
            attn = self.attns[layer_idx]

            # Self-attention with masked softmax
            residual = x
            xn = self.attn_norms[layer_idx](x)

            q = attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)
            k = attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
            v = attn.to_v(xn).reshape(n, h, d).permute(1, 0, 2)

            raw_scores = (q @ k.transpose(-2, -1)) * scale   # (h, N+1, N+1)
            exp_scores = torch.exp(raw_scores)
            exp_masked = exp_scores * col_mask                # pruned cols -> 0
            attn_w = exp_masked * torch.reciprocal(exp_masked.sum(dim=-1, keepdim=True))

            out = (attn_w @ v).permute(1, 0, 2).reshape(n, h * d)
            out = attn.to_out(out)
            x = residual + out

            # Feed-forward block
            x = self.ff_blocks[layer_idx](x)

        # --- Pool + head ---
        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_emb_mask(
    num_tokens: int,
    dim: int,
    kept_patches: set[int],
) -> Tensor:
    """Build embedding mask ``(N+1, D)`` for a pruning decision.

    CLS (index 0) is always kept.
    """
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
# Token classification from zonotope bounds
# ---------------------------------------------------------------------------

def classify_topk(
    ub_scores: Tensor,
    lb_scores: Tensor,
    K: int,
) -> tuple[set[int], set[int], set[int]]:
    """Classify patch tokens as definite-keep / definite-prune / uncertain."""
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


# ---------------------------------------------------------------------------
# Case enumeration
# ---------------------------------------------------------------------------

def enumerate_pruning_cases(
    definite_keep: set[int],
    uncertain: set[int],
    K: int,
) -> list[set[int]]:
    """Enumerate all possible sets of kept patches under the top-K rule."""
    K_remaining = K - len(definite_keep)

    if K_remaining < 0:
        return [definite_keep.copy()]
    if K_remaining >= len(uncertain):
        return [definite_keep | uncertain]
    if K_remaining == 0:
        return [definite_keep.copy()]

    return [
        definite_keep | set(combo)
        for combo in combinations(sorted(uncertain), K_remaining)
    ]


# ---------------------------------------------------------------------------
# Zonotope construction (avoids Cat node)
# ---------------------------------------------------------------------------

def build_zonotope_no_cat(vit, img: Tensor, eps: float, op_patch):
    """Build the embedding zonotope ``(N+1, D)`` without a Cat node."""
    num_patches = (img.shape[-1] // vit.patch_size) ** 2
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()

    patch_zono = op_patch(
        expr.ConstVal(img) + eps * expr.LpEpsilon(list(img.shape))
    )

    pad_op = PadOp(patch_zono.shape, [0, 0, 1, 0])
    padded = AffineSum((pad_op, patch_zono))
    cls_padded = F.pad(vit.cls_token[0], [0, 0, 0, num_patches])
    return padded + ConstVal(cls_padded + vit.pos_embedding[0])


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_scoring(vit, num_tokens: int, dim: int, score_layer: int = 0):
    """Export scoring model, return ``(op_score, scoring_model)``."""
    scoring = ScoringModel(vit, score_layer=score_layer).eval()
    gm = onnx_export(scoring, ([num_tokens + 1, dim],))
    return zono.interpret(gm), scoring


def export_patchify(vit, img_shape: list[int],
                    normalize=False, mean=0.0, std=1.0):
    """Export patchify stage, return ``(op_patch, patchify_model)``."""
    patchify = PatchifyStage(vit, normalize, mean, std).eval()
    gm = onnx_export(patchify, (img_shape,))
    return zono.interpret(gm), patchify


def export_masked_post_concat(vit, emb_mask: Tensor,
                               num_tokens: int, dim: int):
    """Export ``MaskedPostConcat``, return ONNX graph model."""
    model = MaskedPostConcat(vit, emb_mask).eval()
    return onnx_export(model, ([num_tokens + 1, dim],))


# ---------------------------------------------------------------------------
# Per-sample certification
# ---------------------------------------------------------------------------

class CertifyResult(NamedTuple):
    ub: Tensor
    lb: Tensor
    n_cases: int
    definite_keep: set[int]
    definite_prune: set[int]
    uncertain: set[int]


def certify_pruned_sample_diff(
    vit, img: Tensor, eps: float, K: int,
    op_patch, op_score,
    num_tokens: int, dim: int,
) -> CertifyResult:
    """Certify via differential verification: ``full - pruned``.

    Case-splits on uncertain tokens, merges full/pruned models via
    ``diff_net``, propagates with differential zonotope interpreter.
    """
    from boundlab.diff.expr import DiffExpr3
    from boundlab.diff.net import diff_net
    from boundlab.diff.zono3 import interpret as diff_interpret

    full_zono = build_zonotope_no_cat(vit, img, eps, op_patch)

    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    ub_scores, lb_scores = op_score(full_zono).ublb()

    definite_keep, definite_prune, uncertain = classify_topk(ub_scores, lb_scores, K)
    cases = enumerate_pruning_cases(definite_keep, uncertain, K)

    gm_full = export_masked_post_concat(
        vit, build_full_emb_mask(num_tokens, dim), num_tokens, dim,
    )

    best_ub = best_lb = None
    for kept in cases:
        gm_pruned = export_masked_post_concat(
            vit, build_emb_mask(num_tokens, dim, kept), num_tokens, dim,
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