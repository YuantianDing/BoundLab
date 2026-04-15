"""Tests for fusing ``GetSliceOp`` / ``SetSliceOp`` into ``EinsumOp``.

Updated to use the new structured ``list[list[slice]]`` API.
"""

import torch

from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._slicing import GetSliceOp, SetSliceOp
from boundlab.linearop import GetItemOp


def _check_fused(fused, unfused_fwd, unfused_bwd, x, grad,
                 expected_input_shape=None, expected_output_shape=None,
                 atol=1e-5, must_be_einsum=True):
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


def _full(shape, *specs):
    """Build list[list[slice]] from per-dim specs (None = full dim)."""
    result = []
    for d, spec in enumerate(specs):
        if spec is None:
            result.append([slice(0, shape[d])])
        elif isinstance(spec, slice):
            start, stop, _ = spec.indices(shape[d])
            result.append([slice(start, stop)])
        elif isinstance(spec, list):
            result.append(spec)
        else:
            raise ValueError(f"Unexpected spec: {spec}")
    return result


# ========================================================================
# GetSliceOp @ EinsumOp
# ========================================================================


class TestGetSliceFusion:

    def test_slice_batch_output(self):
        """Slice a pure batch output dim -> tensor shrinks."""
        torch.manual_seed(0)
        t = torch.randn(6, 4, 3)
        e = EinsumOp(t, [2], [0, 1])
        gs = GetSliceOp(e.output_shape, _full(e.output_shape, slice(1, 4), None))
        fused = gs @ e
        assert fused.tensor.shape[0] == 3
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(3, 4),
                     expected_output_shape=torch.Size([3, 4]))

    def test_int_index_via_getitem(self):
        """Integer index via GetItemOp drops an output dim."""
        torch.manual_seed(1)
        t = torch.randn(5, 4, 3)
        e = EinsumOp(t, [2], [0, 1])
        gs = GetItemOp(e.output_shape, (2,))
        fused = gs @ e
        assert fused.output_shape == torch.Size([4])
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(4),
                     expected_output_shape=torch.Size([4]),
                     must_be_einsum=False)

    def test_noop_full_slice(self):
        """Full-range slice fuses to an equivalent einsum."""
        torch.manual_seed(4)
        t = torch.randn(5, 3)
        e = EinsumOp(t, [1], [0])
        gs = GetSliceOp(e.output_shape, _full(e.output_shape, None))
        fused = gs @ e
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(5))

    def test_mul_dim_full_slice_ok(self):
        """Mul-dim axis with full slice should still fuse."""
        torch.manual_seed(5)
        t = torch.randn(3, 4)
        e = EinsumOp(t, [0, 1], [0])
        gs = GetSliceOp(e.output_shape, _full(e.output_shape, None))
        fused = gs @ e
        x = torch.randn(3, 4)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(3))

    def test_mul_dim_partial_slice_swaps(self):
        """Non-trivial slice on a mul dim swaps past einsum."""
        from boundlab.linearop._base import ComposedOp
        torch.manual_seed(6)
        t = torch.randn(3, 4)
        e = EinsumOp(t, [0, 1], [0])
        gs = GetSliceOp(e.output_shape, _full(e.output_shape, slice(1, 3)))
        fused = gs @ e
        assert isinstance(fused, ComposedOp)
        x = torch.randn(3, 4)
        y_fused = fused.forward(x)
        y_ref = gs.forward(e.forward(x))
        assert torch.allclose(y_fused, y_ref, atol=1e-5)

    def test_multi_slice_per_dim(self):
        """Multiple non-contiguous slices along one dimension."""
        torch.manual_seed(7)
        t = torch.randn(6, 4, 3)
        e = EinsumOp(t, [2], [0, 1])
        gs = GetSliceOp(e.output_shape, [[slice(0, 2), slice(4, 6)], [slice(0, 4)]])
        fused = gs @ e
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: gs.forward(e.forward(x)),
                     lambda g: e.backward(gs.backward(g)),
                     x, torch.randn(4, 4),
                     expected_output_shape=torch.Size([4, 4]))


# ========================================================================
# EinsumOp @ SetSliceOp
# ========================================================================


class TestSetSliceFusion:

    def test_slice_dot_input(self):
        """Narrowing a pure dot input dim -> tensor shrinks."""
        torch.manual_seed(10)
        t = torch.randn(5, 6)
        e = EinsumOp(t, [1], [0])
        ss = SetSliceOp(torch.Size([6]), [[slice(1, 4)]])
        fused = e @ ss
        assert fused.tensor.shape[1] == 3
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: e.forward(ss.forward(x)),
                     lambda g: ss.backward(e.backward(g)),
                     x, torch.randn(5),
                     expected_input_shape=torch.Size([3]))

    def test_noop_full_slice(self):
        """SetSlice with full slice fuses to equivalent einsum."""
        torch.manual_seed(13)
        t = torch.randn(5, 3)
        e = EinsumOp(t, [1], [0])
        ss = SetSliceOp(torch.Size([3]), [[slice(0, 3)]])
        fused = e @ ss
        x = torch.randn(3)
        _check_fused(fused,
                     lambda x: e.forward(ss.forward(x)),
                     lambda g: ss.backward(e.backward(g)),
                     x, torch.randn(5))

    def test_mul_dim_partial_slice_swaps(self):
        """Slicing a mul-dim input axis swaps past einsum."""
        from boundlab.linearop._base import ComposedOp
        torch.manual_seed(14)
        t = torch.randn(4, 3)
        e = EinsumOp(t, [0, 1], [0])
        ss = SetSliceOp(torch.Size([4, 3]), [[slice(1, 3)], [slice(0, 3)]])
        fused = e @ ss
        assert isinstance(fused, ComposedOp)
        x = torch.randn(2, 3)
        y_fused = fused.forward(x)
        y_ref = e.forward(ss.forward(x))
        assert torch.allclose(y_fused, y_ref, atol=1e-5)

    def test_2d_dot_input(self):
        """Narrowing on a 2D dot input."""
        torch.manual_seed(15)
        t = torch.randn(5, 4, 6)
        e = EinsumOp(t, [1, 2], [0])
        ss = SetSliceOp(torch.Size([4, 6]), [[slice(0, 4)], [slice(1, 4)]])
        fused = e @ ss
        assert fused.tensor.shape[2] == 3
        x = torch.randn(4, 3)
        _check_fused(fused,
                     lambda x: e.forward(ss.forward(x)),
                     lambda g: ss.backward(e.backward(g)),
                     x, torch.randn(5),
                     expected_input_shape=torch.Size([4, 3]))


# ========================================================================
# Slice composition
# ========================================================================


class TestSliceComposition:

    def test_getslice_compose(self):
        """GetSliceOp @ GetSliceOp fuses into one GetSliceOp."""
        torch.manual_seed(20)
        input_shape = torch.Size([10, 8])
        gs1 = GetSliceOp(input_shape, [[slice(2, 8)], [slice(0, 8)]])
        gs2 = GetSliceOp(gs1.output_shape, [[slice(1, 4)], [slice(2, 6)]])
        fused = gs2 @ gs1
        assert isinstance(fused, GetSliceOp)
        x = torch.randn(input_shape)
        y_fused = fused.forward(x)
        y_ref = gs2.forward(gs1.forward(x))
        assert torch.allclose(y_fused, y_ref)


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
        gs = GetSliceOp(e.output_shape, _full(e.output_shape, slice(1, 4), None))
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
        ss = SetSliceOp(torch.Size([6]), [[slice(1, 4)]])
        fused = e @ ss
        x = torch.randn(3)
        self._check_vfwd_vbwd(fused,
                               lambda x: e.forward(ss.forward(x)),
                               lambda g: ss.backward(e.backward(g)),
                               x, torch.randn(5))
