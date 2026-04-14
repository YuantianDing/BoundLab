"""Tests for fusing ``GetSliceOp`` / ``SetSliceOp`` into ``EinsumOp``.

Two fusions are covered:

- ``GetSliceOp @ EinsumOp``: slice output axes -> reduce einsum tensor along
  corresponding output-side tensor dims.
- ``EinsumOp @ SetSliceOp``: embed input into zeros then einsum -> reduce
  tensor along input-side tensor dims.

Each test constructs the composition, checks the fused op is an ``EinsumOp``
with a strictly smaller tensor (when any slice is non-trivial), and verifies
forward / backward match the unfused ``ComposedOp``.
"""

import torch

from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import GetSliceOp, SetSliceOp


def _check_fused(fused, unfused_fwd, unfused_bwd, x, grad,
                 expected_input_shape=None, expected_output_shape=None,
                 atol=1e-5):
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


# ========================================================================
# GetSliceOp @ EinsumOp
# ========================================================================


class TestGetSliceFusion:

    def test_slice_batch_output(self):
        """Slice a pure batch output dim -> tensor shrinks."""
        torch.manual_seed(0)
        t = torch.randn(6, 4, 3)  # output_dims=[0,1], input_dims=[2]
        e = EinsumOp(t, [2], [0, 1])
        gs = GetSliceOp(e.output_shape, (slice(1, 4), slice(None)))
        fused = gs @ e
        assert fused.tensor.shape[0] == 3
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(3, 4),
                     expected_output_shape=torch.Size([3, 4]))

    def test_int_index_drops_output_dim(self):
        """Integer index drops an output dim entirely."""
        torch.manual_seed(1)
        t = torch.randn(5, 4, 3)
        e = EinsumOp(t, [2], [0, 1])
        gs = GetSliceOp(e.output_shape, (2,))  # drop dim 0
        fused = gs @ e
        assert fused.output_shape == torch.Size([4])
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(4),
                     expected_output_shape=torch.Size([4]))

    def test_mixed_slice_and_int(self):
        """Combine an int drop with a slice narrow on another axis."""
        torch.manual_seed(2)
        t = torch.randn(5, 6, 3)
        e = EinsumOp(t, [2], [0, 1])
        gs = GetSliceOp(e.output_shape, (2, slice(1, 4)))
        fused = gs @ e
        assert fused.output_shape == torch.Size([3])
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(3),
                     expected_output_shape=torch.Size([3]))

    def test_ellipsis_and_trailing_slice(self):
        """Ellipsis expands to full-slice on leading dims."""
        torch.manual_seed(3)
        t = torch.randn(4, 5, 3)
        e = EinsumOp(t, [2], [0, 1])
        gs = GetSliceOp(e.output_shape, (Ellipsis, slice(1, 4)))
        fused = gs @ e
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(4, 3),
                     expected_output_shape=torch.Size([4, 3]))

    def test_noop_full_slice(self):
        """Full-range slice on every axis fuses to an equivalent einsum."""
        torch.manual_seed(4)
        t = torch.randn(5, 3)
        e = EinsumOp(t, [1], [0])
        gs = GetSliceOp(e.output_shape, (slice(None),))
        fused = gs @ e
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(5))

    def test_mul_dim_full_slice_ok(self):
        """Mul-dim axis with full slice should still fuse."""
        torch.manual_seed(5)
        # tensor (3, 4): mul dim 0 (appears in input+output), dot dim 1
        t = torch.randn(3, 4)
        e = EinsumOp(t, [0, 1], [0])
        gs = GetSliceOp(e.output_shape, (slice(None),))
        fused = gs @ e
        x = torch.randn(3, 4)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(3))

    def test_mul_dim_partial_slice_bails(self):
        """Non-trivial slice on a mul dim must bail out (ComposedOp)."""
        from boundlab.linearop._base import ComposedOp
        torch.manual_seed(6)
        t = torch.randn(3, 4)
        e = EinsumOp(t, [0, 1], [0])
        gs = GetSliceOp(e.output_shape, (slice(1, 3),))
        fused = gs @ e
        # Should fall back to ComposedOp rather than fusing.
        assert isinstance(fused, ComposedOp)


# ========================================================================
# EinsumOp @ SetSliceOp
# ========================================================================


class TestSetSliceFusion:

    def test_slice_dot_input(self):
        """Narrowing a pure dot input dim -> tensor shrinks on that dim."""
        torch.manual_seed(10)
        # tensor (5, 6): output [0], input [1] (dot)
        t = torch.randn(5, 6)
        e = EinsumOp(t, [1], [0])
        # SetSlice embeds input of shape [3] into zeros of shape [6] at [1:4]
        ss = SetSliceOp((slice(1, 4),),
                        input_shape=torch.Size([3]),
                        output_shape=torch.Size([6]))
        fused = e @ ss
        assert fused.tensor.shape[1] == 3
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: e.forward(ss.forward(x)),
                     lambda g: ss.backward(e.backward(g)),
                     x, torch.randn(5),
                     expected_input_shape=torch.Size([3]))

    def test_int_index_drops_input_dim(self):
        """Integer index on SetSlice drops the input dim."""
        torch.manual_seed(11)
        # tensor (5, 4, 3): output [0], input [1, 2]
        t = torch.randn(5, 4, 3)
        e = EinsumOp(t, [1, 2], [0])
        # SetSlice: input [3] -> output [4, 3], set at [2, :]
        ss = SetSliceOp((2, slice(None)),
                        input_shape=torch.Size([3]),
                        output_shape=torch.Size([4, 3]))
        fused = e @ ss
        assert fused.input_shape == torch.Size([3])
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: e.forward(ss.forward(x)),
                     lambda g: ss.backward(e.backward(g)),
                     x, torch.randn(5),
                     expected_input_shape=torch.Size([3]))

    def test_mixed_int_and_slice(self):
        """Int drop on one axis + slice narrow on another."""
        torch.manual_seed(12)
        # tensor (4, 5, 6): output [0], input [1, 2]
        t = torch.randn(4, 5, 6)
        e = EinsumOp(t, [1, 2], [0])
        ss = SetSliceOp((2, slice(1, 4)),
                        input_shape=torch.Size([3]),
                        output_shape=torch.Size([5, 6]))
        fused = e @ ss
        assert fused.input_shape == torch.Size([3])
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: e.forward(ss.forward(x)),
                     lambda g: ss.backward(e.backward(g)),
                     x, torch.randn(4),
                     expected_input_shape=torch.Size([3]))

    def test_noop_full_slice(self):
        """SetSlice with full slice equals identity; fuses to equivalent einsum."""
        torch.manual_seed(13)
        t = torch.randn(5, 3)
        e = EinsumOp(t, [1], [0])
        ss = SetSliceOp((slice(None),),
                        input_shape=torch.Size([3]),
                        output_shape=torch.Size([3]))
        fused = e @ ss
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: e.forward(ss.forward(x)),
                     lambda g: ss.backward(e.backward(g)),
                     x, torch.randn(5))

    def test_mul_dim_partial_slice_bails(self):
        """Slicing a mul-dim input axis should not fuse."""
        from boundlab.linearop._base import ComposedOp
        torch.manual_seed(14)
        # mul dim: input/output [0], plus dot [1]
        t = torch.randn(4, 3)
        e = EinsumOp(t, [0, 1], [0])
        ss = SetSliceOp((slice(1, 3), slice(None)),
                        input_shape=torch.Size([2, 3]),
                        output_shape=torch.Size([4, 3]))
        fused = e @ ss
        assert isinstance(fused, ComposedOp)


# ========================================================================
# vforward / vbackward on fused ops
# ========================================================================


class TestFusedVBatched:

    def _check_vfwd_vbwd(self, fused, unfused_fwd, unfused_bwd, x, grad,
                          batch_shape=(2, 3), atol=1e-5):
        x_batched = x.unsqueeze(-1).unsqueeze(-1).expand(*x.shape, *batch_shape)
        y_fused = fused.vforward(x_batched)
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                y_ref = unfused_fwd(x_batched[..., i, j])
                assert torch.allclose(y_fused[..., i, j], y_ref, atol=atol)
        g_batched = grad.unsqueeze(0).unsqueeze(0).expand(*batch_shape, *grad.shape)
        dx_fused = fused.vbackward(g_batched)
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                dx_ref = unfused_bwd(g_batched[i, j])
                assert torch.allclose(dx_fused[i, j], dx_ref, atol=atol)

    def test_getslice_vbatched(self):
        torch.manual_seed(30)
        t = torch.randn(6, 4, 3)
        e = EinsumOp(t, [2], [0, 1])
        gs = GetSliceOp(e.output_shape, (slice(1, 4), slice(None)))
        fused = gs @ e
        x = torch.randn(3)
        self._check_vfwd_vbwd(fused,
                               lambda x: gs.forward(e.forward(x)),
                               lambda g: e.backward(gs.backward(g)),
                               x, torch.randn(3, 4))

    def test_setslice_vbatched(self):
        torch.manual_seed(31)
        t = torch.randn(5, 6)
        e = EinsumOp(t, [1], [0])
        ss = SetSliceOp((slice(1, 4),),
                        input_shape=torch.Size([3]),
                        output_shape=torch.Size([6]))
        fused = e @ ss
        x = torch.randn(3)
        self._check_vfwd_vbwd(fused,
                               lambda x: e.forward(ss.forward(x)),
                               lambda g: ss.backward(e.backward(g)),
                               x, torch.randn(5))
