"""Soundness tests for the BoundLab MNIST ViT certifier (split pipeline).

Uses the same Pad+Add strategy as ``certify.py`` to avoid the ``Cat``
expression node, so the original (tight) ``bilinear_matmul`` with
``symmetric_decompose`` works correctly.

Run::

    cd examples/mnist_vit
    pytest tests/test_soundness.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.expr._affine import AffineSum, ConstVal
from boundlab.interp.onnx import onnx_export
from boundlab.linearop import PadOp

from mnist_vit import build_mnist_vit
from certify import PatchifyStage, PostConcatStage, build_zonotope_no_cat

CHECKPOINT = str(_PKG / "mnist_transformer.pt")
FP_ATOL = 1e-4
N_PERTURBATIONS = 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    return build_mnist_vit(CHECKPOINT)


@pytest.fixture(scope="session")
def op_patch(model):
    """Patchify interpreter (pixel → patches)."""
    patchify = PatchifyStage(model).eval()
    gm = onnx_export(patchify, ([1, 28, 28],))
    return zono.interpret(gm)


@pytest.fixture(scope="session")
def op_post(model):
    """Post-concat interpreter (tokens → logits)."""
    post = PostConcatStage(model).eval()
    gm = onnx_export(post, ([17, 64],))
    return zono.interpret(gm)


@pytest.fixture(scope="session")
def sample_images():
    g = torch.Generator().manual_seed(1337)
    return [torch.rand(1, 28, 28, generator=g) for _ in range(3)]


def _certify_one(model, op_patch, op_post, img, eps):
    """Build zonotope via split pipeline and compute bounds."""
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()
    full_zono = build_zonotope_no_cat(
        model, img, eps, op_patch, False, 0.0, 1.0
    )
    return op_post(full_zono).ublb()


# ---------------------------------------------------------------------------
# Adapter correctness
# ---------------------------------------------------------------------------

class TestAdapterCorrectness:

    def test_state_dict_loads_strictly(self):
        vit = build_mnist_vit(None)
        sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        missing, unexpected = vit.load_state_dict(sd, strict=True)
        assert list(missing) == []
        assert list(unexpected) == []

    def test_forward_bit_equivalent_to_deept(self, model):
        import vit_deept
        ref = vit_deept.ViT(
            image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=1, heads=4, mlp_dim=128, layer_norm_type="no_var",
        ).eval()
        ref.load_state_dict(
            torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        )
        torch.manual_seed(0)
        for _ in range(5):
            img = torch.rand(1, 1, 28, 28)
            with torch.no_grad():
                y_ref = ref(img).squeeze(0)
                y_adp = model(img.squeeze(0))
            assert torch.equal(y_ref, y_adp), (
                f"max abs diff = {(y_ref - y_adp).abs().max().item():.3e}"
            )

    def test_adapter_is_no_batch(self, model):
        assert model(torch.rand(1, 28, 28)).shape == (10,)


# ---------------------------------------------------------------------------
# Zonotope invariants
# ---------------------------------------------------------------------------

class TestZonotopeInvariants:

    @pytest.mark.parametrize("eps", [0.0, 1e-5, 1e-3, 1e-2, 5e-2])
    def test_lb_le_ub(self, model, op_patch, op_post, sample_images, eps):
        for img in sample_images:
            ub, lb = _certify_one(model, op_patch, op_post, img, eps)
            assert torch.all(lb <= ub + FP_ATOL)

    def test_zero_eps_encloses_concrete(self, model, op_patch, op_post,
                                         sample_images):
        for img in sample_images:
            ub, lb = _certify_one(model, op_patch, op_post, img, 0.0)
            with torch.no_grad():
                y = model(img)
            assert torch.all(y <= ub + FP_ATOL)
            assert torch.all(y >= lb - FP_ATOL)

    def test_margin_monotone_in_eps(self, model, op_patch, op_post,
                                     sample_images):
        for img in sample_images:
            margins = []
            for eps in [1e-5, 1e-4, 1e-3, 3e-3, 1e-2]:
                ub, lb = _certify_one(model, op_patch, op_post, img, eps)
                pred = int(ub.argmax().item())
                ub_o = ub.clone()
                ub_o[pred] = float("-inf")
                margins.append(float(lb[pred] - ub_o.max()))
            for a, b in zip(margins, margins[1:]):
                assert a + FP_ATOL >= b, f"margin grew: {margins}"


# ---------------------------------------------------------------------------
# Bounds enclosure
# ---------------------------------------------------------------------------

class TestBoundsEnclosure:

    @pytest.mark.parametrize("eps", [1e-3, 5e-3, 1e-2])
    def test_random_perturbations(self, model, op_patch, op_post,
                                   sample_images, eps):
        for idx, img in enumerate(sample_images):
            ub, lb = _certify_one(model, op_patch, op_post, img, eps)
            g = torch.Generator().manual_seed(idx * 31 + int(eps * 1e6))
            for k in range(N_PERTURBATIONS):
                delta = (torch.rand(img.shape, generator=g) * 2 - 1) * eps
                with torch.no_grad():
                    y = model(img + delta)
                assert torch.all(y >= lb - FP_ATOL), (
                    f"sample {idx} eps={eps} trial {k}: "
                    f"below lb by {(lb - y).max():.3e}"
                )
                assert torch.all(y <= ub + FP_ATOL), (
                    f"sample {idx} eps={eps} trial {k}: "
                    f"above ub by {(y - ub).max():.3e}"
                )

    @pytest.mark.parametrize("eps", [1e-3, 5e-3, 1e-2])
    def test_corner_perturbations(self, model, op_patch, op_post,
                                   sample_images, eps):
        for idx, img in enumerate(sample_images):
            ub, lb = _certify_one(model, op_patch, op_post, img, eps)
            g = torch.Generator().manual_seed(idx * 97 + int(eps * 1e6))
            for k in range(20):
                signs = torch.randint(0, 2, img.shape, generator=g) \
                            .float() * 2 - 1
                delta = signs * eps
                with torch.no_grad():
                    y = model(img + delta)
                assert torch.all(y >= lb - FP_ATOL), (
                    f"corner {k} (img {idx}, eps={eps}): "
                    f"below lb by {(lb - y).max():.3e}"
                )
                assert torch.all(y <= ub + FP_ATOL), (
                    f"corner {k} (img {idx}, eps={eps}): "
                    f"above ub by {(y - ub).max():.3e}"
                )


# ---------------------------------------------------------------------------
# PGD cross-check
# ---------------------------------------------------------------------------

def _pgd_attack(model, x, pred, eps, n_steps=50, n_restarts=5):
    alpha = eps / 4
    for _ in range(n_restarts):
        delta = (torch.rand_like(x) * 2 * eps - eps).detach()
        delta.requires_grad_(True)
        for _ in range(n_steps):
            logits = model(x + delta)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[pred] = False
            margin = logits[pred] - logits[mask].max()
            if int(logits.argmax().item()) != pred:
                return True
            grad, = torch.autograd.grad(margin, delta)
            with torch.no_grad():
                delta.sub_(alpha * grad.sign()).clamp_(-eps, eps)
            delta.requires_grad_(True)
    return False


class TestPGDCrosscheck:

    @pytest.mark.parametrize("eps", [1e-3, 3e-3])
    def test_pgd_cannot_break_certified(self, model, op_patch, op_post,
                                         sample_images, eps):
        any_certified = False
        for idx, img in enumerate(sample_images):
            with torch.no_grad():
                pred = int(model(img).argmax().item())
            ub, lb = _certify_one(model, op_patch, op_post, img, eps)
            ub_o = ub.clone()
            ub_o[pred] = float("-inf")
            if float(lb[pred] - ub_o.max()) <= 0:
                continue
            any_certified = True
            flipped = _pgd_attack(model, img, pred, eps)
            assert not flipped, (
                f"SOUNDNESS: PGD flipped certified sample "
                f"(idx={idx}, eps={eps})"
            )
        assert any_certified, "No sample certified; PGD check vacuous."


# ---------------------------------------------------------------------------
# Split-pipeline structure check
# ---------------------------------------------------------------------------

class TestSplitPipeline:

    def test_no_cat_in_expression(self, model, op_patch, sample_images):
        """The split pipeline must produce an AffineSum with symmetric
        LpEpsilon children — no Cat node."""
        img = sample_images[0]
        full_zono = build_zonotope_no_cat(
            model, img, 1e-3, op_patch, False, 0.0, 1.0
        )
        assert isinstance(full_zono, AffineSum)
        assert full_zono.constant is not None, "constant should include cls_token"
        for child in full_zono.children_dict:
            assert child.is_symmetric_to_0(), (
                f"child {type(child).__name__} is not symmetric — "
                f"Cat may still be present"
            )
