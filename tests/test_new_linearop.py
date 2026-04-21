"""Additional tests for refactored LinearOps and their fusion with EinsumOp.

Tests the new structured APIs:
- GetSliceOp/SetSliceOp with list[list[slice]]
- GetIndicesOp/SetIndicesOp with dim-based indexing
- ExpandOp as EinsumOp
- ReshapeOp subclasses (FlattenOp, UnflattenOp, SqueezeOp, UnsqueezeOp)
- Fusion and swapping rules with EinsumOp
"""

import torch
import pytest
from torch.func import vjp

from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._slicing import GetSliceOp, SetSliceOp
from boundlab.linearop._indexing import GetIndicesOp, SetIndicesOp
from boundlab.linearop._reshape import ReshapeOp, FlattenOp, UnflattenOp, SqueezeOp, UnsqueezeOp
from boundlab.linearop._permute import PermuteOp
from boundlab.linearop._expand import ExpandOp
from boundlab.linearop._base import ComposedOp, ScalarOp
from boundlab.linearop import GetItemOp, NarrowOp, SelectOp, PadOp

import boundlab.expr as expr


def check_backward_vs_vjp(op, x, grad, atol=1e-5):
    our = op.backward(grad)
    _, vjp_fn = vjp(op.forward, x)
    (ref,) = vjp_fn(grad)
    assert torch.allclose(our, ref, atol=atol), \
        f"Backward mismatch for {op}: max diff = {(our - ref).abs().max()}"


def check_fwd_bwd(fused, ref_fwd, ref_bwd, x, grad, atol=1e-5):
    y = fused.forward(x)
    y_ref = ref_fwd(x)
    assert torch.allclose(y, y_ref, atol=atol), \
        f"forward mismatch: {(y - y_ref).abs().max()}"
    dx = fused.backward(grad)
    dx_ref = ref_bwd(grad)
    assert torch.allclose(dx, dx_ref, atol=atol), \
        f"backward mismatch: {(dx - dx_ref).abs().max()}"


# ========================================================================
# ExpandOp as EinsumOp
# ========================================================================

class TestExpandOpIsEinsum:
    def test_same_ndim_is_einsum(self):
        op = ExpandOp(torch.Size([1, 3, 1]), torch.Size([4, 3, 5]))
        assert isinstance(op, EinsumOp)
        x = torch.randn(1, 3, 1)
        y = op.forward(x)
        assert y.shape == torch.Size([4, 3, 5])
        assert torch.allclose(y, x.expand(4, 3, 5))

    def test_noop_returns_scalar(self):
        op = ExpandOp(torch.Size([3, 4]), torch.Size([3, 4]))
        assert isinstance(op, ScalarOp)

    def test_extra_dims_still_works(self):
        op = ExpandOp(torch.Size([3]), (2, 3))
        x = torch.randn(3)
        y = op.forward(x)
        assert y.shape == torch.Size([2, 3])
        assert torch.allclose(y, x.expand(2, 3))

    def test_backward_correctness(self):
        op = ExpandOp(torch.Size([1, 4]), torch.Size([3, 4]))
        x = torch.randn(1, 4)
        grad = torch.randn(3, 4)
        check_backward_vs_vjp(op, x, grad)

    def test_backward_extra_dims(self):
        op = ExpandOp(torch.Size([4]), (2, 4))
        x = torch.randn(4)
        grad = torch.randn(2, 4)
        check_backward_vs_vjp(op, x, grad)


# ========================================================================
# ReshapeOp subclasses
# ========================================================================

class TestReshapeSubclasses:
    def test_flatten_is_reshape(self):
        op = FlattenOp(torch.Size([2, 3, 4]), 1, 2)
        assert isinstance(op, ReshapeOp)

    def test_unflatten_is_reshape(self):
        op = UnflattenOp(torch.Size([2, 12]), 1, (3, 4))
        assert isinstance(op, ReshapeOp)

    def test_squeeze_is_reshape(self):
        op = SqueezeOp(torch.Size([2, 1, 4]), 1)
        assert isinstance(op, ReshapeOp)

    def test_unsqueeze_is_reshape(self):
        op = UnsqueezeOp(torch.Size([2, 4]), 1)
        assert isinstance(op, ReshapeOp)

    def test_flatten_fuses_with_einsum(self):
        torch.manual_seed(0)
        e = EinsumOp(torch.randn(3, 6), [1], [0])  # input=[6], output=[3]
        f = FlattenOp(torch.Size([2, 3]), 0, 1)     # input=[2,3], output=[6]
        fused = e @ f
        assert isinstance(fused, EinsumOp)
        x = torch.randn(2, 3)
        check_fwd_bwd(fused,
                      lambda x: e.forward(f.forward(x)),
                      lambda g: f.backward(e.backward(g)),
                      x, torch.randn(3))


# ========================================================================
# GetSliceOp new API
# ========================================================================

class TestNewGetSliceOp:
    def test_single_slice_per_dim(self):
        op = GetSliceOp(torch.Size([10, 8]), [[slice(2, 5)], [slice(1, 6)]])
        assert op.output_shape == torch.Size([3, 5])
        x = torch.randn(10, 8)
        y = op.forward(x)
        assert torch.allclose(y, x[2:5, 1:6])

    def test_multi_slice(self):
        op = GetSliceOp(torch.Size([10]), [[slice(0, 2), slice(5, 8)]])
        assert op.output_shape == torch.Size([5])
        x = torch.arange(10, dtype=torch.float)
        y = op.forward(x)
        assert torch.allclose(y, torch.tensor([0., 1., 5., 6., 7.]))

    def test_backward_single(self):
        op = GetSliceOp(torch.Size([5, 4]), [[slice(1, 3)], [slice(0, 4)]])
        x = torch.randn(5, 4)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_backward_multi(self):
        op = GetSliceOp(torch.Size([8, 4]), [[slice(0, 2), slice(5, 7)], [slice(0, 4)]])
        x = torch.randn(8, 4)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_compose_two_getslice(self):
        op1 = GetSliceOp(torch.Size([10, 8]), [[slice(1, 7)], [slice(0, 8)]])
        op2 = GetSliceOp(op1.output_shape, [[slice(2, 5)], [slice(1, 4)]])
        fused = op2 @ op1
        assert isinstance(fused, GetSliceOp)
        x = torch.randn(10, 8)
        assert torch.allclose(fused.forward(x), op2.forward(op1.forward(x)))


# ========================================================================
# SetSliceOp new API
# ========================================================================

class TestNewSetSliceOp:
    def test_basic(self):
        op = SetSliceOp(torch.Size([5, 4]), [[slice(1, 3)], [slice(0, 4)]])
        assert op.input_shape == torch.Size([2, 4])
        x = torch.randn(2, 4)
        y = op.forward(x)
        assert y.shape == torch.Size([5, 4])
        assert torch.allclose(y[1:3], x)
        assert (y[0] == 0).all() and (y[3] == 0).all() and (y[4] == 0).all()

    def test_backward(self):
        op = SetSliceOp(torch.Size([5, 4]), [[slice(1, 3)], [slice(0, 4)]])
        x = torch.randn(op.input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


# ========================================================================
# GetIndicesOp new API
# ========================================================================

class TestNewGetIndicesOp:
    def test_1d_indices(self):
        indices = torch.tensor([3, 1, 0])
        op = GetIndicesOp(torch.Size([5, 4]), dim=0, indices=indices, added_shape=torch.Size([3]))
        assert op.output_shape == torch.Size([3, 4])
        x = torch.randn(5, 4)
        y = op.forward(x)
        assert torch.allclose(y[0], x[3])
        assert torch.allclose(y[1], x[1])
        assert torch.allclose(y[2], x[0])

    def test_backward(self):
        indices = torch.tensor([2, 0, 4, 1])
        op = GetIndicesOp(torch.Size([5, 3]), dim=0, indices=indices, added_shape=torch.Size([4]))
        x = torch.randn(5, 3)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_dim1(self):
        indices = torch.tensor([1, 3])
        op = GetIndicesOp(torch.Size([4, 5, 3]), dim=1, indices=indices, added_shape=torch.Size([2]))
        assert op.output_shape == torch.Size([4, 2, 3])
        x = torch.randn(4, 5, 3)
        y = op.forward(x)
        assert torch.allclose(y[:, 0, :], x[:, 1, :])
        assert torch.allclose(y[:, 1, :], x[:, 3, :])

    def test_compose_getindices(self):
        """GetIndicesOp @ GetIndicesOp on same dim composes."""
        idx1 = torch.tensor([3, 1, 4, 0, 2])
        op1 = GetIndicesOp(torch.Size([6, 3]), dim=0, indices=idx1, added_shape=torch.Size([5]))
        idx2 = torch.tensor([0, 2])
        op2 = GetIndicesOp(op1.output_shape, dim=0, indices=idx2, added_shape=torch.Size([2]))
        fused = op2 @ op1
        assert isinstance(fused, GetIndicesOp)
        x = torch.randn(6, 3)
        assert torch.allclose(fused.forward(x), op2.forward(op1.forward(x)))


# ========================================================================
# SetIndicesOp new API
# ========================================================================

class TestNewSetIndicesOp:
    def test_basic(self):
        indices = torch.tensor([2, 0])
        op = SetIndicesOp(torch.Size([5, 3]), dim=0, indices=indices, added_shape=torch.Size([2]))
        assert op.input_shape == torch.Size([2, 3])
        x = torch.randn(2, 3)
        y = op.forward(x)
        assert y.shape == torch.Size([5, 3])
        assert torch.allclose(y[2], x[0])
        assert torch.allclose(y[0], x[1])

    def test_backward(self):
        indices = torch.tensor([1, 3, 0])
        op = SetIndicesOp(torch.Size([5, 4]), dim=0, indices=indices, added_shape=torch.Size([3]))
        x = torch.randn(op.input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


# ========================================================================
# GetIndicesOp / SetIndicesOp fusion with EinsumOp
# ========================================================================

class TestIndicesFusion:
    def test_getindices_dot_dim(self):
        """GetIndicesOp on a dot/batch dim fuses into EinsumOp."""
        torch.manual_seed(0)
        t = torch.randn(5, 3)
        e = EinsumOp(t, [1], [0])  # output dim 0 is batch
        idx = torch.tensor([0, 3, 4])
        gi = GetIndicesOp(e.output_shape, dim=0, indices=idx, added_shape=torch.Size([3]))
        fused = gi @ e
        assert isinstance(fused, EinsumOp)
        x = torch.randn(3)
        check_fwd_bwd(fused,
                      lambda x: gi.forward(e.forward(x)),
                      lambda g: e.backward(gi.backward(g)),
                      x, torch.randn(3))

    def test_setindices_dot_dim(self):
        """EinsumOp @ SetIndicesOp on a dot dim fuses."""
        torch.manual_seed(1)
        t = torch.randn(4, 6)
        e = EinsumOp(t, [1], [0])  # input dim is dot
        idx = torch.tensor([0, 2, 5])
        si = SetIndicesOp(torch.Size([6]), dim=0, indices=idx, added_shape=torch.Size([3]))
        fused = e @ si
        assert isinstance(fused, EinsumOp)
        x = torch.randn(3)
        check_fwd_bwd(fused,
                      lambda x: e.forward(si.forward(x)),
                      lambda g: si.backward(e.backward(g)),
                      x, torch.randn(4))

    def test_getindices_mul_dim_swaps(self):
        """GetIndicesOp on a mul dim swaps past EinsumOp."""
        torch.manual_seed(2)
        t = torch.randn(4, 3)
        e = EinsumOp(t, [0, 1], [0])  # dim 0 is mul
        idx = torch.tensor([0, 2])
        gi = GetIndicesOp(e.output_shape, dim=0, indices=idx, added_shape=torch.Size([2]))
        fused = gi @ e
        assert isinstance(fused, ComposedOp)
        x = torch.randn(4, 3)
        y_fused = fused.forward(x)
        y_ref = gi.forward(e.forward(x))
        assert torch.allclose(y_fused, y_ref, atol=1e-5)

    def test_setindices_mul_dim_swaps(self):
        """EinsumOp @ SetIndicesOp on a mul dim swaps."""
        torch.manual_seed(3)
        t = torch.randn(4, 3)
        e = EinsumOp(t, [0, 1], [0])  # input=[4,3], output=[4], dim 0 is mul
        idx = torch.tensor([0, 2])
        si = SetIndicesOp(torch.Size([4, 3]), dim=0, indices=idx, added_shape=torch.Size([2]))
        fused = e @ si
        assert isinstance(fused, ComposedOp)
        x = torch.randn(2, 3)
        y_fused = fused.forward(x)
        y_ref = e.forward(si.forward(x))
        assert torch.allclose(y_fused, y_ref, atol=1e-5)


# ========================================================================
# GetSliceOp / SetSliceOp fusion with EinsumOp (additional)
# ========================================================================

class TestSliceFusionAdditional:
    def test_getslice_batch_dim_fuses(self):
        torch.manual_seed(10)
        t = torch.randn(8, 3)
        e = EinsumOp(t, [1], [0])
        gs = GetSliceOp(e.output_shape, [[slice(2, 6)]])
        fused = gs @ e
        assert isinstance(fused, EinsumOp)
        x = torch.randn(3)
        check_fwd_bwd(fused,
                      lambda x: gs.forward(e.forward(x)),
                      lambda g: e.backward(gs.backward(g)),
                      x, torch.randn(4))

    def test_setslice_dot_dim_fuses(self):
        torch.manual_seed(11)
        t = torch.randn(4, 8)
        e = EinsumOp(t, [1], [0])
        ss = SetSliceOp(torch.Size([8]), [[slice(1, 5)]])
        fused = e @ ss
        assert isinstance(fused, EinsumOp)
        x = torch.randn(4)
        check_fwd_bwd(fused,
                      lambda x: e.forward(ss.forward(x)),
                      lambda g: ss.backward(e.backward(g)),
                      x, torch.randn(4))


# ========================================================================
# Convenience aliases
# ========================================================================

class TestConvenienceAliases:
    def test_narrow(self):
        op = NarrowOp(torch.Size([5, 4]), dim=0, start=1, length=3)
        assert isinstance(op, GetSliceOp)
        x = torch.randn(5, 4)
        assert torch.allclose(op.forward(x), x[1:4])

    def test_select(self):
        op = SelectOp(torch.Size([5, 4]), dim=0, index=2)
        x = torch.randn(5, 4)
        assert torch.allclose(op.forward(x), x[2])

    def test_getitem_slice(self):
        op = GetItemOp(torch.Size([5, 4]), (slice(1, 3), slice(None)))
        x = torch.randn(5, 4)
        assert torch.allclose(op.forward(x), x[1:3])

    def test_getitem_int(self):
        op = GetItemOp(torch.Size([5, 4, 3]), (2, slice(1, 3)))
        x = torch.randn(5, 4, 3)
        assert torch.allclose(op.forward(x), x[2, 1:3])

    def test_pad(self):
        op = PadOp(torch.Size([3]), [1, 2])
        assert isinstance(op, SetSliceOp)
        x = torch.tensor([1., 2., 3.])
        y = op.forward(x)
        assert torch.allclose(y, torch.tensor([0., 1., 2., 3., 0., 0.]))

    def test_select_backward(self):
        op = SelectOp(torch.Size([5, 4]), dim=0, index=2)
        x = torch.randn(5, 4)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_getitem_backward(self):
        op = GetItemOp(torch.Size([5, 4, 3]), (2, slice(1, 3)))
        x = torch.randn(5, 4, 3)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


# ========================================================================
# Expr API integration
# ========================================================================

class TestExprIntegration:
    def test_expand_with_new_dims(self):
        center = expr.ConstVal(torch.randn(1, 3))
        eps = expr.LpEpsilon([1, 3])
        x = expr.Add(center, eps)
        y = x.expand(2, 1, 3)
        ub, lb = y.ublb()
        assert ub.shape == torch.Size([2, 1, 3])

    def test_expand_on(self):
        center = expr.ConstVal(torch.randn(1, 3, 1))
        eps = expr.LpEpsilon([1, 3, 1])
        x = expr.Add(center, eps)
        y = x.expand_on(2, 5)
        ub, lb = y.ublb()
        assert ub.shape == torch.Size([1, 3, 5])

    def test_chain_slice_expand(self):
        center = expr.ConstVal(torch.randn(4, 1, 6))
        eps = expr.LpEpsilon([4, 1, 6])
        x = expr.Add(center, eps)
        y = x.narrow(2, 1, 3).expand_on(1, 5)
        ub, lb = y.ublb()
        assert ub.shape == torch.Size([4, 5, 3])
