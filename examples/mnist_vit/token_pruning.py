"""Token pruning logic: scoring, classification, case enumeration, and certification.

Workflow
-------
1. ``build_input_zonotope`` — build symbolic zonotope from image + perturbation.
2. ``export_scoring`` / ``op_score(zonotope)`` — get interval-bounded importance scores.
3. ``classify_topk`` — partition tokens into definite-keep / definite-prune / uncertain.
4. ``enumerate_pruning_cases`` — list all valid kept-sets from uncertain tokens.
5. Per case: ``export_pruned_vit`` → ``diff_net`` → ``diff_interpret`` → union bounds.

Or use ``certify_pruning_diff`` for the full pipeline.
"""
from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.expr._affine import AffineSum, ConstVal
from boundlab.interp.onnx import onnx_export
from boundlab.linearop import PadOp

from pipeline import ScoringModel, PrunedViT


# ---------------------------------------------------------------------------
# Score / mask construction
# ---------------------------------------------------------------------------

def build_token_scores(num_tokens: int, kept_patches: set[int],
                       magnitude: float = 100.0) -> Tensor:
    """Build per-token scores for Heaviside pruning.

    Returns a ``(num_tokens + 1,)`` tensor with ``+magnitude`` for kept
    tokens and ``-magnitude`` for pruned tokens.  CLS (index 0) is always
    kept.

    With ``magnitude=100``, the Heaviside linearizer sees definite
    positive/negative bounds → exact 0/1 mask with zero approximation error.
    """
    scores = torch.full((num_tokens + 1,), -magnitude)
    scores[0] = magnitude                   # CLS always kept
    for p in kept_patches:
        scores[p + 1] = magnitude
    return scores


def build_all_kept_scores(num_tokens: int, magnitude: float = 100.0) -> Tensor:
    """All tokens kept (no pruning).  Equivalent to an all-ones mask."""
    return build_token_scores(num_tokens, set(range(num_tokens)), magnitude)


# Legacy mask builders (used by pruning_zono.py and old tests)

def build_token_mask(num_tokens: int, dim: int, kept_patches: set[int]) -> Tensor:
    """Build ``(N+1, D)`` binary mask.  CLS always kept."""
    total = num_tokens + 1
    mask = torch.zeros(total, dim)
    mask[0] = 1.0
    for p in kept_patches:
        mask[p + 1] = 1.0
    return mask


def build_full_token_mask(num_tokens: int, dim: int) -> Tensor:
    """All-ones mask (no pruning)."""
    return build_token_mask(num_tokens, dim, set(range(num_tokens)))


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

def classify_topk(ub_scores: Tensor, lb_scores: Tensor, K: int):
    """Partition N tokens into definite-keep / definite-prune / uncertain.

    A token is *definite-keep* if its lower bound beats the upper bound of
    at least ``N - K`` other tokens (it's guaranteed top-K).  A token is
    *definite-prune* if its upper bound is beaten by the lower bound of at
    least K other tokens (it's guaranteed not top-K).

    Returns:
        ``(definite_keep, definite_prune, uncertain)`` — three sets of
        patch indices (0-based, CLS excluded).
    """
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


def enumerate_pruning_cases(definite_keep: set[int], uncertain: set[int],
                            K: int) -> list[set[int]]:
    """Generate all valid kept-sets by choosing from uncertain tokens.

    Each returned set has exactly ``K`` patch indices (or fewer if
    ``definite_keep`` already has ``>= K``).
    """
    K_remaining = K - len(definite_keep)
    if K_remaining < 0:
        return [definite_keep.copy()]
    if K_remaining >= len(uncertain):
        return [definite_keep | uncertain]
    if K_remaining == 0:
        return [definite_keep.copy()]
    return [definite_keep | set(c)
            for c in combinations(sorted(uncertain), K_remaining)]


# ---------------------------------------------------------------------------
# Zonotope construction
# ---------------------------------------------------------------------------

def build_input_zonotope(vit, img: Tensor, eps: float, op_patch):
    """Build the symbolic zonotope for the full token sequence.

    Runs the pixel image through the patch embedding interpreter, prepends
    CLS token via ``PadOp`` (avoids a ``Cat`` node that would break
    ``symmetric_decompose``), and adds positional embedding.

    Args:
        vit: The ViT model (needs ``patch_size``, ``cls_token``, ``pos_embedding``).
        img: Input image, shape ``(C, H, W)``.
        eps: L∞ perturbation radius in pixel space.
        op_patch: Zonotope interpreter for ``vit.to_patch_embedding``.

    Returns:
        An ``AffineSum`` expression representing the uncertain token
        sequence ``(N+1, D)``.
    """
    num_patches = (img.shape[-1] // vit.patch_size) ** 2
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    patch_zono = op_patch(expr.ConstVal(img) + eps * expr.LpEpsilon(list(img.shape)))
    pad_op = PadOp(patch_zono.shape, [0, 0, 1, 0])
    padded = AffineSum((pad_op, patch_zono))
    cls_padded = F.pad(vit.cls_token[0], [0, 0, 0, num_patches])
    return padded + ConstVal(cls_padded + vit.pos_embedding[0])


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def export_patch_embedding(vit, img_shape: list[int]):
    """ONNX-export ``vit.to_patch_embedding``; return zonotope interpreter."""
    gm = onnx_export(vit.to_patch_embedding, (img_shape,))
    return zono.interpret(gm)


def export_scoring(vit, num_tokens: int, dim: int, score_layer: int = 0):
    """ONNX-export ``ScoringModel``; return ``(interpreter, concrete_model)``."""
    scoring = ScoringModel(vit, score_layer=score_layer).eval()
    gm = onnx_export(scoring, ([num_tokens + 1, dim],))
    return zono.interpret(gm), scoring


def export_pruned_vit(vit, kept_patches: set[int], num_tokens: int, dim: int,
                      mask_from_layer: int = 0) -> object:
    """ONNX-export a ``PrunedViT`` with Heaviside ops for verification.

    Args:
        vit: Source ViT model.
        kept_patches: Set of patch indices to keep.
        num_tokens: Number of patch tokens (excluding CLS).
        dim: Embedding dimension.
        mask_from_layer: First layer to apply pruning.

    Returns:
        ONNX IR model ready for ``zono.interpret`` or ``diff_net``.
    """
    scores = build_token_scores(num_tokens, kept_patches)
    model = PrunedViT(vit, scores, mask_from_layer=mask_from_layer,
                      for_verification=True).eval()
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


def certify_pruning_diff(
    vit, img: Tensor, eps: float, K: int, op_patch, op_score,
    num_tokens: int, dim: int, mask_from_layer: int = 0,
) -> CertifyResult:
    """End-to-end differential certification of token pruning.

    1. Build zonotope from image + perturbation.
    2. Score tokens → classify top-K → enumerate cases.
    3. Per case: diff_net(full, pruned) → diff_interpret → union bounds.

    Returns a ``CertifyResult`` with the worst-case ub/lb over all cases.
    """
    from boundlab.diff.expr import DiffExpr3
    from boundlab.diff.net import diff_net
    from boundlab.diff.zono3 import interpret as diff_interpret_fn

    full_zono = build_input_zonotope(vit, img, eps, op_patch)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    ub_scores, lb_scores = op_score(full_zono).ublb()
    definite_keep, definite_prune, uncertain = classify_topk(ub_scores, lb_scores, K)
    cases = enumerate_pruning_cases(definite_keep, uncertain, K)

    gm_full = export_pruned_vit(vit, set(range(num_tokens)),
                                num_tokens, dim, mask_from_layer)

    best_ub = best_lb = None
    for kept in cases:
        gm_pruned = export_pruned_vit(vit, kept, num_tokens, dim, mask_from_layer)
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        merged = diff_net(gm_full, gm_pruned)
        out = diff_interpret_fn(merged)(full_zono)
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