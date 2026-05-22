"""Test that Heaviside-based pruning produces identical bounds to concrete masks.

With ±large token scores, ``heaviside_pruning`` should give exact 0/1 with
zero linearization error — mathematically equivalent to multiplying by a
concrete binary mask.

Run::

    cd examples/mnist_vit
    pytest tests/test_heaviside_equivalence.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mnist_vit import build_mnist_vit
from pipeline import ScoringModel, PrunedViT
from token_pruning import (
    build_token_scores, build_all_kept_scores,
    build_input_zonotope, export_patch_embedding,
    export_pruned_vit,
)
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.diff.expr import DiffExpr3

CHECKPOINT = str(_PKG / "mnist_transformer.pt")
FP_ATOL = 1e-5


@pytest.fixture(scope="session")
def vit():
    return build_mnist_vit(CHECKPOINT)


@pytest.fixture(scope="session")
def sample_image():
    g = torch.Generator().manual_seed(42)
    return torch.rand(1, 28, 28, generator=g)


# ---------------------------------------------------------------------------
# Concrete forward equivalence
# ---------------------------------------------------------------------------

class TestConcreteForward:
    """Verify that PrunedViT (for_verification=False) produces the same
    output as the original ViT when no tokens are pruned."""

    def test_full_mask_matches_vit(self, vit, sample_image):
        scores = build_all_kept_scores(16)
        pruned = PrunedViT(vit, scores, for_verification=False).eval()
        with torch.no_grad():
            y_vit = vit(sample_image)
            emb = vit.to_patch_embedding(sample_image)
            tokens = torch.cat((vit.cls_token[0], emb), dim=0) + vit.pos_embedding[0]
            y_pruned = pruned(tokens)
        # Note: may differ slightly because PrunedViT decomposes softmax
        # differently (pairwise diff → exp → mask → sum → reciprocal vs
        # standard softmax). Check within tolerance.
        assert torch.allclose(y_vit, y_pruned, atol=1e-4), (
            f"max diff = {(y_vit - y_pruned).abs().max():.3e}"
        )

    def test_pruned_zeros_tokens(self, vit, sample_image):
        """With some tokens pruned, verify pruned token embeddings are zero."""
        kept = {0, 1, 2, 3, 4, 5, 6, 7}
        scores = build_token_scores(16, kept)
        pruned = PrunedViT(vit, scores, for_verification=False).eval()
        with torch.no_grad():
            emb = vit.to_patch_embedding(sample_image)
            tokens = torch.cat((vit.cls_token[0], emb), dim=0) + vit.pos_embedding[0]

            # Hook into _apply_token_mask to check the masked output
            original_mask = pruned._apply_token_mask
            masked_result = [None]
            def capture_mask(x):
                result = original_mask(x)
                masked_result[0] = result
                return result
            pruned._apply_token_mask = capture_mask
            _ = pruned(tokens)

            # Pruned tokens (indices 9-16, i.e. patches 8-15) should be zero
            for p in range(16):
                if p not in kept:
                    assert masked_result[0][p + 1].abs().max() == 0, (
                        f"Pruned token {p} not zeroed"
                    )


# ---------------------------------------------------------------------------
# Zonotope bound equivalence
# ---------------------------------------------------------------------------

class TestZonoBoundEquivalence:
    """Verify that Heaviside zonotope bounds enclose the concrete output
    (same as the soundness test but through the Heaviside path)."""

    @pytest.mark.parametrize("eps", [1e-3, 5e-3])
    def test_heaviside_bounds_enclose_concrete(self, vit, sample_image, eps):
        """Random perturbations must stay within zonotope bounds."""
        kept = {0, 1, 2, 3, 4, 5, 6, 7}
        op_patch = export_patch_embedding(vit, [1, 28, 28])

        # Build zonotope and compute bounds through PrunedViT
        full_zono = build_input_zonotope(vit, sample_image, eps, op_patch)
        gm = export_pruned_vit(vit, kept, 16, 64)
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        out = zono.interpret(gm)(full_zono)
        ub, lb = out.ublb()

        # Verify concrete perturbations stay within bounds
        model = PrunedViT(
            vit, build_token_scores(16, kept), for_verification=False,
        ).eval()
        g = torch.Generator().manual_seed(123)
        for _ in range(100):
            delta = (torch.rand(sample_image.shape, generator=g) * 2 - 1) * eps
            perturbed = sample_image + delta
            with torch.no_grad():
                emb = vit.to_patch_embedding(perturbed)
                tokens = (torch.cat((vit.cls_token[0], emb), dim=0)
                          + vit.pos_embedding[0])
                y = model(tokens)
            assert torch.all(y >= lb - FP_ATOL), (
                f"below lb by {(lb - y).max():.3e}"
            )
            assert torch.all(y <= ub + FP_ATOL), (
                f"above ub by {(y - ub).max():.3e}"
            )


# ---------------------------------------------------------------------------
# Differential bound equivalence
# ---------------------------------------------------------------------------

class TestDiffBoundEquivalence:
    """Verify diff_net with Heaviside-based PrunedViT produces sound bounds."""

    @pytest.mark.parametrize("eps", [1e-3, 5e-3])
    def test_diff_bounds_enclose_mc(self, vit, sample_image, eps):
        """MC max|full - pruned| ≤ diff bound."""
        kept = {0, 1, 2, 3, 4, 5, 6, 7}
        op_patch = export_patch_embedding(vit, [1, 28, 28])

        # Diff bound
        full_zono = build_input_zonotope(vit, sample_image, eps, op_patch)
        gm_full = export_pruned_vit(vit, set(range(16)), 16, 64)
        gm_pruned = export_pruned_vit(vit, kept, 16, 64)
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        merged = diff_net(gm_full, gm_pruned)
        out = diff_interpret(merged)(full_zono)
        if isinstance(out, DiffExpr3):
            d_ub, d_lb = out.diff.ublb()
        else:
            d_ub, d_lb = (out.x - out.y).ublb()
        diff_bound = max(d_ub.abs().max().item(), d_lb.abs().max().item())

        # MC bound — perturb in pixel space to match verification
        model_full = PrunedViT(
            vit, build_all_kept_scores(16), for_verification=False,
        ).eval()
        model_pruned = PrunedViT(
            vit, build_token_scores(16, kept), for_verification=False,
        ).eval()
        mc_max = 0.0
        g = torch.Generator().manual_seed(456)
        with torch.no_grad():
            for _ in range(200):
                delta = (torch.rand(sample_image.shape, generator=g) * 2 - 1) * eps
                img_p = sample_image + delta
                emb = vit.to_patch_embedding(img_p)
                xp = (torch.cat((vit.cls_token[0], emb), dim=0)
                      + vit.pos_embedding[0])
                diff = model_full(xp) - model_pruned(xp)
                mc_max = max(mc_max, diff.abs().max().item())

        assert mc_max <= diff_bound + FP_ATOL, (
            f"MC {mc_max:.6f} > diff bound {diff_bound:.6f}"
        )