"""Tests for fusing shape LinearOps (Reshape/Expand/Squeeze/Unsqueeze) into EinsumOp.

Each test constructs an EinsumOp and a shape op, fuses them via ``@``, and
verifies that the fused EinsumOp produces identical forward / backward results
compared to the unfused ComposedOp.
"""

import torch
import pytest

from boundlab.linearop._base import ComposedOp
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._shape import (
    ReshapeOp,
    ExpandOp,
    SqueezeOp,
    UnsqueezeOp,
    PermuteOp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_fused(fused, unfused_fwd, unfused_bwd,
                 x: torch.Tensor, grad: torch.Tensor,
                 expected_input_shape=None, expected_output_shape=None,
                 atol=1e-5, must_be_einsum=True):
    """Compare fused op against unfused forward / backward callables."""
    if must_be_einsum:
        assert isinstance(fused, EinsumOp), \
            f"expected EinsumOp, got {type(fused).__name__}: {fused}"
    if expected_input_shape is not None:
        assert fused.input_shape == expected_input_shape, \
            f"input_shape: {fused.input_shape} != {expected_input_shape}"
    if expected_output_shape is not None:
        assert fused.output_shape == expected_output_shape, \
            f"output_shape: {fused.output_shape} != {expected_output_shape}"

    y_fused = fused.forward(x)
    y_ref = unfused_fwd(x)
    assert torch.allclose(y_fused, y_ref, atol=atol), \
        f"forward mismatch: max diff = {(y_fused - y_ref).abs().max()}"

    dx_fused = fused.backward(grad)
    dx_ref = unfused_bwd(grad)
    assert torch.allclose(dx_fused, dx_ref, atol=atol), \
        f"backward mismatch: max diff = {(dx_fused - dx_ref).abs().max()}"


def _make_einsum(tensor_shape, input_dims, output_dims, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(tensor_shape)
    return EinsumOp(t, list(input_dims), list(output_dims))


# ========================================================================
# Unsqueeze × EinsumOp
# ========================================================================


class TestUnsqueezeFusion:
    """EinsumOp @ UnsqueezeOp and UnsqueezeOp @ EinsumOp."""

    def test_input_dot(self):
        """Unsqueeze adds a size-1 dim inside a dot-only input."""
        e = _make_einsum((5, 3, 1, 4), [1, 2, 3], [0])
        u = UnsqueezeOp(torch.Size([3, 4]), dim=1)
        fused = e @ u
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: e.forward(u.forward(x)),
                      lambda g: u.backward(e.backward(g)),
                      x, torch.randn(5),
                      expected_input_shape=torch.Size([3, 4]),
                      expected_output_shape=torch.Size([5]))

    def test_input_leading(self):
        """Unsqueeze at dim 0."""
        e = _make_einsum((1, 4, 3), [0, 1], [2])
        u = UnsqueezeOp(torch.Size([4]), dim=0)
        fused = e @ u
        x = torch.randn(4)
        _check_fused(fused, lambda x: e.forward(u.forward(x)),
                      lambda g: u.backward(e.backward(g)),
                      x, torch.randn(3),
                      expected_input_shape=torch.Size([4]),
                      expected_output_shape=torch.Size([3]))

    def test_input_trailing(self):
        """Unsqueeze at last dim."""
        e = _make_einsum((3, 4, 1, 2), [0, 1, 2], [3])
        u = UnsqueezeOp(torch.Size([3, 4]), dim=2)
        fused = e @ u
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: e.forward(u.forward(x)),
                      lambda g: u.backward(e.backward(g)),
                      x, torch.randn(2))

    def test_output_leading(self):
        """Unsqueeze at dim 0 on the output side."""
        e = _make_einsum((5, 3, 4), [1, 2], [0])
        u = UnsqueezeOp(e.output_shape, dim=0)
        fused = u @ e
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: u.forward(e.forward(x)),
                      lambda g: e.backward(u.backward(g)),
                      x, torch.randn(1, 5),
                      expected_output_shape=torch.Size([1, 5]))

    def test_output_trailing(self):
        """Unsqueeze appending a dim after the last output position."""
        e = _make_einsum((5, 3, 4), [1, 2], [0])
        u = UnsqueezeOp(e.output_shape, dim=1)
        fused = u @ e
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: u.forward(e.forward(x)),
                      lambda g: e.backward(u.backward(g)),
                      x, torch.randn(5, 1),
                      expected_output_shape=torch.Size([5, 1]))

    def test_input_mul_dim(self):
        """Unsqueeze inside a Hadamard einsum (mul dims present)."""
        # tensor (3, 4): both dims are mul. input=(3,1,4), output=(3,4).
        # Unsqueeze (3, 4) -> (3, 1, 4).
        e = _make_einsum((3, 1, 4), [0, 1, 2], [0, 2])
        u = UnsqueezeOp(torch.Size([3, 4]), dim=1)
        fused = e @ u
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: e.forward(u.forward(x)),
                      lambda g: u.backward(e.backward(g)),
                      x, torch.randn(3, 4))


# ========================================================================
# Squeeze × EinsumOp
# ========================================================================


class TestSqueezeFusion:
    """EinsumOp @ SqueezeOp and SqueezeOp @ EinsumOp."""

    def test_input_single_dim(self):
        """Squeeze removes one size-1 dim from input."""
        e = _make_einsum((5, 3, 4), [1, 2], [0])
        s = SqueezeOp(torch.Size([3, 1, 4]), dim=1)
        fused = e @ s
        x = torch.randn(3, 1, 4)
        _check_fused(fused, lambda x: e.forward(s.forward(x)),
                      lambda g: s.backward(e.backward(g)),
                      x, torch.randn(5),
                      expected_input_shape=torch.Size([3, 1, 4]))

    def test_input_all_dims(self):
        """Squeeze(dim=None) removes multiple size-1 dims."""
        e = _make_einsum((5, 3, 4), [1, 2], [0])
        s = SqueezeOp(torch.Size([1, 3, 1, 4, 1]), dim=None)
        fused = e @ s
        x = torch.randn(1, 3, 1, 4, 1)
        _check_fused(fused, lambda x: e.forward(s.forward(x)),
                      lambda g: s.backward(e.backward(g)),
                      x, torch.randn(5),
                      expected_input_shape=torch.Size([1, 3, 1, 4, 1]))

    def test_output_single_dim(self):
        """Squeeze removes a size-1 dim from output."""
        e = _make_einsum((1, 5, 3, 4), [2, 3], [0, 1])
        s = SqueezeOp(e.output_shape, dim=0)
        fused = s @ e
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: s.forward(e.forward(x)),
                      lambda g: e.backward(s.backward(g)),
                      x, torch.randn(5),
                      expected_output_shape=torch.Size([5]))

    def test_output_all_dims(self):
        """Squeeze(dim=None) removes all size-1 output dims."""
        e = _make_einsum((1, 5, 1, 3), [3], [0, 1, 2])
        s = SqueezeOp(e.output_shape, dim=None)
        fused = s @ e
        x = torch.randn(3)
        _check_fused(fused, lambda x: s.forward(e.forward(x)),
                      lambda g: e.backward(s.backward(g)),
                      x, torch.randn(5),
                      expected_output_shape=torch.Size([5]))

    def test_noop(self):
        """Squeeze on a dim that isn't size-1 is a no-op."""
        e = _make_einsum((5, 3, 4), [1, 2], [0])
        s = SqueezeOp(torch.Size([3, 4]), dim=0)  # size 3, not 1 → noop
        fused = e @ s
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: e.forward(s.forward(x)),
                      lambda g: s.backward(e.backward(g)),
                      x, torch.randn(5))


# ========================================================================
# Reshape × EinsumOp
# ========================================================================


class TestReshapeFusion:
    """EinsumOp @ ReshapeOp and ReshapeOp @ EinsumOp (no mul dims)."""

    def test_input_merge(self):
        """Merge two input dims into one."""
        e = _make_einsum((5, 6, 4), [1, 2], [0])
        r = ReshapeOp(torch.Size([2, 3, 4]), (6, 4))
        fused = e @ r
        x = torch.randn(2, 3, 4)
        _check_fused(fused, lambda x: e.forward(r.forward(x)),
                      lambda g: r.backward(e.backward(g)),
                      x, torch.randn(5),
                      expected_input_shape=torch.Size([2, 3, 4]))

    def test_input_split(self):
        """Split one input dim into two."""
        e = _make_einsum((5, 12), [1], [0])
        r = ReshapeOp(torch.Size([3, 4]), (12,))
        fused = e @ r
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: e.forward(r.forward(x)),
                      lambda g: r.backward(e.backward(g)),
                      x, torch.randn(5),
                      expected_input_shape=torch.Size([3, 4]))

    def test_output_merge(self):
        """Merge two output dims into one."""
        e = _make_einsum((6, 4, 3), [2], [0, 1])
        r = ReshapeOp(torch.Size([6, 4]), (24,))
        fused = r @ e
        x = torch.randn(3)
        _check_fused(fused, lambda x: r.forward(e.forward(x)),
                      lambda g: e.backward(r.backward(g)),
                      x, torch.randn(24),
                      expected_output_shape=torch.Size([24]))

    def test_output_split(self):
        """Split one output dim into several."""
        e = _make_einsum((6, 4, 4), [2], [0, 1])
        r = ReshapeOp(torch.Size([6, 4]), (2, 3, 4))
        fused = r @ e
        x = torch.randn(4)
        _check_fused(fused, lambda x: r.forward(e.forward(x)),
                      lambda g: e.backward(r.backward(g)),
                      x, torch.randn(2, 3, 4),
                      expected_output_shape=torch.Size([2, 3, 4]))


class TestReshapeFusionWithMulDims:
    """Reshape fusion when EinsumOp has mul_dims (Hadamard-style dims).

    The reshape must leave mul_dim positions intact (1-to-1 size match) and
    only restructure pure dot / batch dims.
    """

    def test_input_dot_only_reshaped(self):
        """Mul dim at front; reshape only touches trailing dot dims."""
        torch.manual_seed(7)
        t = torch.randn(3, 6, 3)
        e = EinsumOp(t, input_dims=[0, 1, 2], output_dims=[0])
        r = ReshapeOp(torch.Size([3, 2, 3, 3]), (3, 6, 3))
        fused = e @ r
        x = torch.randn(3, 2, 3, 3)
        _check_fused(fused, lambda x: e.forward(r.forward(x)),
                      lambda g: r.backward(e.backward(g)),
                      x, torch.randn(3))

    def test_output_batch_only_reshaped(self):
        """Mul dim at back; reshape only touches leading batch dims."""
        torch.manual_seed(8)
        t = torch.randn(4, 6, 4)
        e = EinsumOp(t, input_dims=[2], output_dims=[0, 1, 2])
        r = ReshapeOp(torch.Size([4, 6, 4]), (2, 2, 6, 4))
        fused = r @ e
        x = torch.randn(4)
        _check_fused(fused, lambda x: r.forward(e.forward(x)),
                      lambda g: e.backward(r.backward(g)),
                      x, torch.randn(2, 2, 6, 4))

    def test_input_mul_in_middle(self):
        """Mul dim sandwiched between two dot runs."""
        torch.manual_seed(9)
        t = torch.randn(2, 3, 4)
        e = EinsumOp(t, input_dims=[0, 1, 2], output_dims=[1])
        r = ReshapeOp(torch.Size([2, 3, 2, 2]), (2, 3, 4))
        fused = e @ r
        x = torch.randn(2, 3, 2, 2)
        _check_fused(fused, lambda x: e.forward(r.forward(x)),
                      lambda g: r.backward(e.backward(g)),
                      x, torch.randn(3))

    def test_output_mul_in_middle(self):
        """Mul dim between two batch runs that get reshaped."""
        torch.manual_seed(10)
        t = torch.randn(6, 3, 4)
        e = EinsumOp(t, input_dims=[1], output_dims=[0, 1, 2])
        r = ReshapeOp(torch.Size([6, 3, 4]), (2, 3, 3, 2, 2))
        fused = r @ e
        x = torch.randn(3)
        _check_fused(fused, lambda x: r.forward(e.forward(x)),
                      lambda g: e.backward(r.backward(g)),
                      x, torch.randn(2, 3, 3, 2, 2))

    def test_multiple_mul_dims(self):
        """Two mul dims with dot dims between and around them."""
        torch.manual_seed(11)
        # tensor (2, 3, 4, 5): dims 1,3 are mul.
        # input_dims = [0, 1, 2, 3], output_dims = [1, 3].
        t = torch.randn(2, 3, 4, 5)
        e = EinsumOp(t, input_dims=[0, 1, 2, 3], output_dims=[1, 3])
        # Reshape (2, 3, 2, 2, 5) -> (2, 3, 4, 5). Only dim 2 changes: (2,2)->4.
        r = ReshapeOp(torch.Size([2, 3, 2, 2, 5]), (2, 3, 4, 5))
        fused = e @ r
        x = torch.randn(2, 3, 2, 2, 5)
        _check_fused(fused, lambda x: e.forward(r.forward(x)),
                      lambda g: r.backward(e.backward(g)),
                      x, torch.randn(3, 5))


# ========================================================================
# Expand × EinsumOp
# ========================================================================


class TestExpandFusion:
    """EinsumOp @ ExpandOp and ExpandOp @ EinsumOp — batch/dot dims."""

    def test_output_leading(self):
        """Add new leading dim on output side."""
        e = _make_einsum((4, 3), [1], [0])
        ex = ExpandOp(e.output_shape, (2, 4))
        fused = ex @ e
        x = torch.randn(3)
        _check_fused(fused, lambda x: ex.forward(e.forward(x)),
                      lambda g: e.backward(ex.backward(g)),
                      x, torch.randn(2, 4),
                      expected_output_shape=torch.Size([2, 4]))

    def test_output_broadcast(self):
        """Broadcast a size-1 batch dim on output side."""
        e = _make_einsum((1, 4, 3), [2], [0, 1])
        ex = ExpandOp(e.output_shape, (5, 4))
        fused = ex @ e
        x = torch.randn(3)
        _check_fused(fused, lambda x: ex.forward(e.forward(x)),
                      lambda g: e.backward(ex.backward(g)),
                      x, torch.randn(5, 4))

    def test_input_leading(self):
        """Expand adds a new leading dim on input side."""
        e = _make_einsum((4, 2, 3), [1, 2], [0])
        ex = ExpandOp(torch.Size([3]), (2, 3))
        fused = e @ ex
        x = torch.randn(3)
        _check_fused(fused, lambda x: e.forward(ex.forward(x)),
                      lambda g: ex.backward(e.backward(g)),
                      x, torch.randn(4))

    def test_input_broadcast(self):
        """Expand broadcasts a size-1 input dim."""
        e = _make_einsum((4, 2, 3), [1, 2], [0])
        ex = ExpandOp(torch.Size([1, 3]), (2, 3))
        fused = e @ ex
        x = torch.randn(1, 3)
        _check_fused(fused, lambda x: e.forward(ex.forward(x)),
                      lambda g: ex.backward(e.backward(g)),
                      x, torch.randn(4))


class TestExpandFusionWithMulDims:
    """Expand fusion where the expanded dim is also a mul_dim."""

    def test_input_mul_broadcast(self):
        """Broadcast a size-1 input dim that is also a mul dim.

        einsum has mul dim (size k in tensor, size k in input/output).
        Expand broadcasts input from 1→k.  Fused op should produce correct
        results with the smaller (1,…) input.
        """
        torch.manual_seed(20)
        # tensor (3, 4): input_dims=[0,1], output_dims=[0] → dim 0 is mul.
        t = torch.randn(3, 4)
        e = EinsumOp(t, input_dims=[0, 1], output_dims=[0])
        # Expand (1, 4) → (3, 4).
        ex = ExpandOp(torch.Size([1, 4]), (3, 4))
        fused = e @ ex
        x = torch.randn(1, 4)
        _check_fused(fused, lambda x: e.forward(ex.forward(x)),
                      lambda g: ex.backward(e.backward(g)),
                      x, torch.randn(3),
                      expected_input_shape=torch.Size([1, 4]))

    def test_input_mul_broadcast_multiple(self):
        """Two mul dims, one of which is broadcast."""
        torch.manual_seed(21)
        # tensor (3, 4, 5): input_dims=[0,1,2], output_dims=[0,2].
        # Mul dims = [0, 2]. Expand from (1, 4, 5) → (3, 4, 5).
        t = torch.randn(3, 4, 5)
        e = EinsumOp(t, input_dims=[0, 1, 2], output_dims=[0, 2])
        ex = ExpandOp(torch.Size([1, 4, 5]), (3, 4, 5))
        fused = e @ ex
        x = torch.randn(1, 4, 5)
        _check_fused(fused, lambda x: e.forward(ex.forward(x)),
                      lambda g: ex.backward(e.backward(g)),
                      x, torch.randn(3, 5),
                      expected_input_shape=torch.Size([1, 4, 5]))

    def test_input_mul_leading_and_broadcast(self):
        """Expand adds new leading dim AND broadcasts a mul dim."""
        torch.manual_seed(22)
        # tensor (3, 4): input_dims=[0,1], output_dims=[0].
        # Expand (4,) → (3, 4): new leading dim + same shape.
        t = torch.randn(3, 4)
        e = EinsumOp(t, input_dims=[0, 1], output_dims=[0])
        ex = ExpandOp(torch.Size([4]), (3, 4))
        fused = e @ ex
        x = torch.randn(4)
        _check_fused(fused, lambda x: e.forward(ex.forward(x)),
                      lambda g: ex.backward(e.backward(g)),
                      x, torch.randn(3),
                      expected_input_shape=torch.Size([4]))

    def test_output_mul_broadcast(self):
        """Broadcast a size-1 output dim that is also a mul dim."""
        torch.manual_seed(23)
        # tensor (1, 4): input_dims=[0,1], output_dims=[0,1] → all mul.
        # output = (1, 4). Expand (1, 4) → (3, 4).
        t = torch.randn(1, 4)
        e = EinsumOp(t, input_dims=[0, 1], output_dims=[0, 1])
        ex = ExpandOp(e.output_shape, (3, 4))
        fused = ex @ e
        x = torch.randn(1, 4)
        _check_fused(fused, lambda x: ex.forward(e.forward(x)),
                      lambda g: e.backward(ex.backward(g)),
                      x, torch.randn(3, 4),
                      expected_output_shape=torch.Size([3, 4]))

    def test_output_mul_broadcast_leading_and_existing(self):
        """New leading dim + broadcast of an existing mul dim on output."""
        torch.manual_seed(24)
        # tensor (1, 3): input_dims=[0,1], output_dims=[0,1].
        # Expand (1, 3) → (2, 5, 3): new leading dim 2, broadcast 1→5.
        t = torch.randn(1, 3)
        e = EinsumOp(t, input_dims=[0, 1], output_dims=[0, 1])
        ex = ExpandOp(e.output_shape, (2, 5, 3))
        fused = ex @ e
        x = torch.randn(1, 3)
        _check_fused(fused, lambda x: ex.forward(e.forward(x)),
                      lambda g: e.backward(ex.backward(g)),
                      x, torch.randn(2, 5, 3),
                      expected_output_shape=torch.Size([2, 5, 3]))


# ========================================================================
# Chained fusions — multiple ops fused sequentially
# ========================================================================


class TestChainedFusion:
    """Verify that multiple shape ops fuse sequentially into a single EinsumOp."""

    def test_reshape_then_unsqueeze_input(self):
        """(einsum @ reshape @ unsqueeze) should fuse into one EinsumOp."""
        e = _make_einsum((5, 6), [1], [0])
        r = ReshapeOp(torch.Size([2, 3]), (6,))
        u = UnsqueezeOp(torch.Size([2, 3]), dim=0)
        step1 = e @ r
        assert isinstance(step1, EinsumOp)
        fused = step1 @ u
        assert isinstance(fused, EinsumOp)
        x = torch.randn(2, 3)
        # Compare to unfused chain
        ref_fwd = lambda x: e.forward(r.forward(u.forward(x)))
        ref_bwd = lambda g: u.backward(r.backward(e.backward(g)))
        _check_fused(fused, ref_fwd, ref_bwd, x, torch.randn(5))

    def test_squeeze_then_reshape_output(self):
        """(squeeze @ reshape @ einsum) fused step by step."""
        e = _make_einsum((6, 4, 3), [2], [0, 1])
        r = ReshapeOp(torch.Size([6, 4]), (24,))
        s = SqueezeOp(torch.Size([1, 24]), dim=0)  # won't fire — 24 != 1
        # Use a real squeeze: insert unsqueeze first to get a size-1 dim
        u = UnsqueezeOp(torch.Size([24]), dim=0)
        step1 = r @ e
        assert isinstance(step1, EinsumOp)
        step2 = u @ step1
        assert isinstance(step2, EinsumOp)
        s2 = SqueezeOp(step2.output_shape, dim=0)
        fused = s2 @ step2
        assert isinstance(fused, EinsumOp)
        x = torch.randn(3)
        ref_fwd = lambda x: s2.forward(u.forward(r.forward(e.forward(x))))
        ref_bwd = lambda g: e.backward(r.backward(u.backward(s2.backward(g))))
        _check_fused(fused, ref_fwd, ref_bwd, x, torch.randn(24))

    def test_expand_then_reshape_output(self):
        """(reshape @ expand @ einsum) chained."""
        e = _make_einsum((4, 3), [1], [0])
        ex = ExpandOp(e.output_shape, (2, 4))
        r = ReshapeOp(torch.Size([2, 4]), (8,))
        step1 = ex @ e
        assert isinstance(step1, EinsumOp)
        fused = r @ step1
        assert isinstance(fused, EinsumOp)
        x = torch.randn(3)
        ref_fwd = lambda x: r.forward(ex.forward(e.forward(x)))
        ref_bwd = lambda g: e.backward(ex.backward(r.backward(g)))
        _check_fused(fused, ref_fwd, ref_bwd, x, torch.randn(8))

    def test_permute_then_reshape_then_expand(self):
        """(expand @ reshape @ permute @ einsum) on output side."""
        e = _make_einsum((3, 4, 5), [2], [0, 1])
        p = PermuteOp(e.output_shape, (1, 0))
        r = ReshapeOp(p.output_shape, (12,))
        ex = ExpandOp(r.output_shape, (2, 12))

        step1 = p @ e
        assert isinstance(step1, EinsumOp)
        step2 = r @ step1
        assert isinstance(step2, EinsumOp)
        fused = ex @ step2
        assert isinstance(fused, EinsumOp)
        x = torch.randn(5)
        ref_fwd = lambda x: ex.forward(r.forward(p.forward(e.forward(x))))
        ref_bwd = lambda g: e.backward(p.backward(r.backward(ex.backward(g))))
        _check_fused(fused, ref_fwd, ref_bwd, x, torch.randn(2, 12))


# ========================================================================
# Parametric stress tests
# ========================================================================


@pytest.mark.parametrize("seed", range(5))
class TestParametricFusion:
    """Random-seed parametric tests to catch edge cases."""

    def test_reshape_roundtrip_input(self, seed):
        """Reshape input → fuse → check (various seeds)."""
        torch.manual_seed(seed)
        t = torch.randn(7, 12)
        e = EinsumOp(t, [1], [0])
        r = ReshapeOp(torch.Size([3, 4]), (12,))
        fused = e @ r
        x = torch.randn(3, 4)
        _check_fused(fused, lambda x: e.forward(r.forward(x)),
                      lambda g: r.backward(e.backward(g)),
                      x, torch.randn(7))

    def test_reshape_roundtrip_output(self, seed):
        """Reshape output → fuse → check (various seeds)."""
        torch.manual_seed(seed)
        t = torch.randn(12, 5)
        e = EinsumOp(t, [1], [0])
        r = ReshapeOp(torch.Size([12]), (3, 4))
        fused = r @ e
        x = torch.randn(5)
        _check_fused(fused, lambda x: r.forward(e.forward(x)),
                      lambda g: e.backward(r.backward(g)),
                      x, torch.randn(3, 4))

    def test_expand_broadcast_input(self, seed):
        """Expand broadcast on input side (various seeds)."""
        torch.manual_seed(seed)
        t = torch.randn(5, 3, 4)
        e = EinsumOp(t, [1, 2], [0])
        ex = ExpandOp(torch.Size([1, 4]), (3, 4))
        fused = e @ ex
        x = torch.randn(1, 4)
        _check_fused(fused, lambda x: e.forward(ex.forward(x)),
                      lambda g: ex.backward(e.backward(g)),
                      x, torch.randn(5))

    def test_expand_broadcast_output(self, seed):
        """Expand broadcast on output side (various seeds)."""
        torch.manual_seed(seed)
        t = torch.randn(1, 3, 4)
        e = EinsumOp(t, [2], [0, 1])
        ex = ExpandOp(e.output_shape, (5, 3))
        fused = ex @ e
        x = torch.randn(4)
        _check_fused(fused, lambda x: ex.forward(e.forward(x)),
                      lambda g: e.backward(ex.backward(g)),
                      x, torch.randn(5, 3))

    def test_unsqueeze_squeeze_inverse(self, seed):
        """Unsqueeze then squeeze should be identity."""
        torch.manual_seed(seed)
        t = torch.randn(3, 4)
        e = EinsumOp(t, [1], [0])
        u = UnsqueezeOp(e.output_shape, dim=0)
        s = SqueezeOp(u.output_shape, dim=0)
        step1 = u @ e
        fused = s @ step1
        assert isinstance(fused, EinsumOp)
        x = torch.randn(4)
        _check_fused(fused, lambda x: e.forward(x),
                      lambda g: e.backward(g),
                      x, torch.randn(3))


# ========================================================================
# vforward / vbackward consistency
# ========================================================================


class TestFusedVForwardVBackward:
    """Verify vforward/vbackward of the fused EinsumOp match batched unfused."""

    def _check_vfwd_vbwd(self, fused, unfused_fwd, unfused_bwd, x, grad,
                          batch_shape=(2, 3), atol=1e-5):
        # vforward: x has trailing batch dims
        x_batched = x.unsqueeze(-1).unsqueeze(-1).expand(*x.shape, *batch_shape)
        y_fused = fused.vforward(x_batched)
        # reference per-element
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                y_ref = unfused_fwd(x_batched[..., i, j])
                assert torch.allclose(y_fused[..., i, j], y_ref, atol=atol)
        # vbackward: grad has leading batch dims
        g_batched = grad.unsqueeze(0).unsqueeze(0).expand(*batch_shape, *grad.shape)
        dx_fused = fused.vbackward(g_batched)
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                dx_ref = unfused_bwd(g_batched[i, j])
                assert torch.allclose(dx_fused[i, j], dx_ref, atol=atol)

    def test_reshape_input_vfwd(self):
        e = _make_einsum((5, 6, 4), [1, 2], [0])
        r = ReshapeOp(torch.Size([2, 3, 4]), (6, 4))
        fused = e @ r
        x = torch.randn(2, 3, 4)
        self._check_vfwd_vbwd(fused,
                               lambda x: e.forward(r.forward(x)),
                               lambda g: r.backward(e.backward(g)),
                               x, torch.randn(5))

    def test_expand_output_vfwd(self):
        e = _make_einsum((1, 4, 3), [2], [0, 1])
        ex = ExpandOp(e.output_shape, (5, 4))
        fused = ex @ e
        x = torch.randn(3)
        self._check_vfwd_vbwd(fused,
                               lambda x: ex.forward(e.forward(x)),
                               lambda g: e.backward(ex.backward(g)),
                               x, torch.randn(5, 4))

    def test_unsqueeze_output_vfwd(self):
        e = _make_einsum((5, 3, 4), [1, 2], [0])
        u = UnsqueezeOp(e.output_shape, dim=0)
        fused = u @ e
        x = torch.randn(3, 4)
        self._check_vfwd_vbwd(fused,
                               lambda x: u.forward(e.forward(x)),
                               lambda g: e.backward(u.backward(g)),
                               x, torch.randn(1, 5))

    def test_expand_mul_input_vfwd(self):
        """vforward/vbackward for mul-dim expand fusion."""
        torch.manual_seed(20)
        t = torch.randn(3, 4)
        e = EinsumOp(t, input_dims=[0, 1], output_dims=[0])
        ex = ExpandOp(torch.Size([1, 4]), (3, 4))
        fused = e @ ex
        x = torch.randn(1, 4)
        self._check_vfwd_vbwd(fused,
                               lambda x: e.forward(ex.forward(x)),
                               lambda g: ex.backward(e.backward(g)),
                               x, torch.randn(3))
