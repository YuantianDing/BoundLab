"""Tests for shape and indices LinearOp implementations.

Tests verify:
1. backward() matches torch.func.vjp for correctness
2. vforward/vbackward work correctly with batched inputs
3. ublb() returns sound bounds when applied to expressions
"""

import torch
import pytest
from torch.func import vjp

import boundlab.expr as expr
from boundlab.linearop import (
    # Shape ops
    ReshapeOp,
    FlattenOp,
    UnflattenOp,
    PermuteOp,
    TransposeOp,
    SqueezeOp,
    UnsqueezeOp,
    ExpandOp,
    RepeatOp,
    TileOp,
    FlipOp,
    RollOp,
    DiagOp,
    # Indexing ops
    GatherOp,
    ScatterOp,
    GetIndicesOp,
    SetIndicesOp,
    GetSliceOp,
    SetSliceOp,
    # Convenience aliases
    NarrowOp,
    SelectOp,
    GetItemOp,
    PadOp,
)


# ============================================================================
# Helper functions
# ============================================================================


def check_backward_vs_vjp(op, x: torch.Tensor, grad: torch.Tensor, atol: float = 1e-5):
    """Verify op.backward(grad) matches VJP computation."""
    # Compute using our backward
    our_backward = op.backward(grad)

    # Compute using VJP
    _, vjp_fn = vjp(op.forward, x)
    (vjp_backward,) = vjp_fn(grad)

    assert torch.allclose(our_backward, vjp_backward, atol=atol), (
        f"Backward mismatch for {op}:\n"
        f"  our_backward: {our_backward}\n"
        f"  vjp_backward: {vjp_backward}\n"
        f"  max diff: {(our_backward - vjp_backward).abs().max()}"
    )


def check_vforward_vbackward(op, x: torch.Tensor, batch_shape: tuple, atol: float = 1e-5):
    """Verify vforward and vbackward are consistent with batched operations."""
    # Create batched input for vforward (trailing batch dims)
    x_batched = x.unsqueeze(-1).unsqueeze(-1).expand(*x.shape, *batch_shape)

    # vforward
    y_batched = op.vforward(x_batched)
    assert y_batched.shape == (*op.output_shape, *batch_shape), (
        f"vforward shape mismatch: expected {(*op.output_shape, *batch_shape)}, got {y_batched.shape}"
    )

    # Verify each batch element matches unbatched forward
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            y_single = op.forward(x_batched[..., i, j])
            assert torch.allclose(y_batched[..., i, j], y_single, atol=atol), (
                f"vforward result mismatch at batch [{i}, {j}]"
            )

    # Create batched grad for vbackward (leading batch dims)
    grad = torch.randn(op.output_shape)
    grad_batched = grad.unsqueeze(0).unsqueeze(0).expand(*batch_shape, *grad.shape)

    # vbackward
    dx_batched = op.vbackward(grad_batched)
    assert dx_batched.shape == (*batch_shape, *op.input_shape), (
        f"vbackward shape mismatch: expected {(*batch_shape, *op.input_shape)}, got {dx_batched.shape}"
    )

    # Verify each batch element matches unbatched backward
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            dx_single = op.backward(grad_batched[i, j])
            assert torch.allclose(dx_batched[i, j], dx_single, atol=atol), (
                f"vbackward result mismatch at batch [{i}, {j}]"
            )


def _sample_inputs(center: torch.Tensor, n: int = 2000) -> torch.Tensor:
    """Uniform samples from the L∞ ball of radius 1 around `center`."""
    eps = torch.rand(n, *center.shape) * 2 - 1
    return center.unsqueeze(0) + eps


def _check_bounds(
    outputs: torch.Tensor, ub: torch.Tensor, lb: torch.Tensor, tol: float = 1e-5
):
    """Assert all outputs lie within [lb, ub]."""
    assert (outputs <= ub.unsqueeze(0) + tol).all(), (
        f"Upper bound violated: max excess = {(outputs - ub.unsqueeze(0)).max():.6f}"
    )
    assert (outputs >= lb.unsqueeze(0) - tol).all(), (
        f"Lower bound violated: max deficit = {(lb.unsqueeze(0) - outputs).max():.6f}"
    )


def _make_input(center_val: torch.Tensor):
    """Create an expression: center + eps where eps ∈ [-1, 1]."""
    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon(list(center_val.shape))
    return expr.Add(center, eps), center_val


# ============================================================================
# Backward vs VJP tests for Shape Ops
# ============================================================================


class TestReshapeOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        target_shape = (6, 4)
        op = ReshapeOp(input_shape, target_shape)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_vforward_vbackward(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = ReshapeOp(input_shape, (6, 4))
        x = torch.randn(input_shape)
        check_vforward_vbackward(op, x, (3, 2))

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3, 4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.reshape(6, 4)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.reshape(-1, 6, 4)
        _check_bounds(outputs, ub, lb)


class TestFlattenOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4, 5])
        op = FlattenOp(input_shape, start_dim=1, end_dim=2)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3, 4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.flatten(0, 1)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.flatten(1, 2)
        _check_bounds(outputs, ub, lb)


class TestUnflattenOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 12, 5])
        op = UnflattenOp(input_shape, dim=1, sizes=(3, 4))

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 12)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.unflatten(1, (3, 4))

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        # samples has batch dim at 0, so unflatten dim 2 (was dim 1 in expr)
        outputs = samples.unflatten(2, (3, 4))
        _check_bounds(outputs, ub, lb)


class TestPermuteOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = PermuteOp(input_shape, dims=(2, 0, 1))

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_vforward_vbackward(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = PermuteOp(input_shape, dims=(2, 0, 1))
        x = torch.randn(input_shape)
        check_vforward_vbackward(op, x, (3, 2))

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3, 4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.permute(2, 0, 1)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.permute(0, 3, 1, 2)  # batch dim is 0
        _check_bounds(outputs, ub, lb)


class TestTransposeOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = TransposeOp(input_shape, dim0=0, dim1=2)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_vforward_vbackward(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = TransposeOp(input_shape, dim0=0, dim1=2)
        x = torch.randn(input_shape)
        check_vforward_vbackward(op, x, (3, 2))

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3, 4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.transpose(0, 2)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.transpose(1, 3)  # account for batch dim
        _check_bounds(outputs, ub, lb)


class TestSqueezeOp:
    def test_backward_vs_vjp_with_dim(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 1, 4])
        op = SqueezeOp(input_shape, dim=1)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_backward_vs_vjp_all_dims(self):
        torch.manual_seed(42)
        input_shape = torch.Size([1, 3, 1, 4, 1])
        op = SqueezeOp(input_shape, dim=None)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 1, 4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.squeeze(1)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.squeeze(2)  # account for batch dim
        _check_bounds(outputs, ub, lb)


class TestUnsqueezeOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = UnsqueezeOp(input_shape, dim=1)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_vforward_vbackward(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = UnsqueezeOp(input_shape, dim=1)
        x = torch.randn(input_shape)
        check_vforward_vbackward(op, x, (3, 2))

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3, 4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.unsqueeze(1)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.unsqueeze(2)  # account for batch dim
        _check_bounds(outputs, ub, lb)


class TestExpandOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([1, 3, 1])
        sizes = (4, 3, 5)
        op = ExpandOp(input_shape, sizes)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(1, 3, 1)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.expand(4, 3, 5)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.expand(-1, 4, 3, 5)
        _check_bounds(outputs, ub, lb)


class TestRepeatOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3])
        sizes = (3, 2)
        op = RepeatOp(input_shape, sizes)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.repeat(2, 3)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.repeat(1, 2, 3)  # batch dim doesn't repeat
        _check_bounds(outputs, ub, lb)


class TestTileOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        sizes = (2, 1, 3)
        op = TileOp(input_shape, sizes)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.tile(2, 3)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.tile(1, 2, 3)
        _check_bounds(outputs, ub, lb)


class TestFlipOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = FlipOp(input_shape, dims=(0, 2))

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3, 4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.flip((0, 2))

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.flip((1, 3))  # account for batch dim
        _check_bounds(outputs, ub, lb)


class TestRollOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])
        op = RollOp(input_shape, shifts=(1, -2), dims=(0, 2))

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(2, 3, 4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.roll(2, dims=1)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.roll(2, dims=2)  # account for batch dim
        _check_bounds(outputs, ub, lb)


class TestDiagOp:
    def test_backward_vs_vjp_1d_to_2d(self):
        torch.manual_seed(42)
        input_shape = torch.Size([5])
        op = DiagOp(input_shape, diagonal=0)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_backward_vs_vjp_2d_to_1d(self):
        torch.manual_seed(42)
        input_shape = torch.Size([4, 6])
        op = DiagOp(input_shape, diagonal=1)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness_1d(self):
        torch.manual_seed(42)
        center_val = torch.randn(4)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.diag()

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = torch.stack([s.diag() for s in samples])
        _check_bounds(outputs, ub, lb)


# ============================================================================
# Backward vs VJP tests for Indexing Ops
# ============================================================================


class TestNarrowOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([5, 8, 10])
        op = NarrowOp(input_shape, dim=1, start=2, length=4)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(5, 8)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr.narrow(1, 2, 4)

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples.narrow(2, 2, 4)  # account for batch dim
        _check_bounds(outputs, ub, lb)


class TestSelectOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([5, 8, 10])
        op = SelectOp(input_shape, dim=1, index=3)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


class TestGetItemOp:
    def test_backward_vs_vjp_simple_slice(self):
        torch.manual_seed(42)
        input_shape = torch.Size([5, 8, 10])
        indices = (slice(1, 4), slice(None), slice(2, 8, 2))
        op = GetItemOp(input_shape, indices)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_backward_vs_vjp_int_index(self):
        torch.manual_seed(42)
        input_shape = torch.Size([5, 8, 10])
        indices = (2, slice(1, 5), slice(None))
        op = GetItemOp(input_shape, indices)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_ublb_soundness(self):
        torch.manual_seed(42)
        center_val = torch.randn(5, 8, 10)
        x_expr, center = _make_input(center_val)
        y_expr = x_expr[1:4, :, 2:8]

        ub, lb = y_expr.ublb()
        samples = _sample_inputs(center)
        outputs = samples[:, 1:4, :, 2:8]
        _check_bounds(outputs, ub, lb)


class TestPadOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([3, 4, 5])
        pad_spec = [1, 2, 0, 1]  # pad last 2 dims
        op = PadOp(input_shape, pad_spec)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


class TestGetSliceOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([5, 8, 10])
        indices = (slice(1, 4), 3, slice(2, 8))
        op = GetSliceOp(input_shape, indices)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


class TestSetSliceOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([3, 6])
        output_shape = torch.Size([5, 8])
        indices = (slice(1, 4), slice(1, 7))
        op = SetSliceOp(indices, input_shape, output_shape)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


class TestGatherOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([4, 5, 6])
        dim = 1
        index = torch.randint(0, 5, (4, 3, 6))
        op = GatherOp(input_shape, dim, index)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_vforward_vbackward(self):
        torch.manual_seed(42)
        input_shape = torch.Size([4, 5, 6])
        dim = 1
        index = torch.randint(0, 5, (4, 3, 6))
        op = GatherOp(input_shape, dim, index)
        x = torch.randn(input_shape)
        check_vforward_vbackward(op, x, (2, 3))


class TestScatterOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([4, 3, 6])
        output_shape = torch.Size([4, 5, 6])
        dim = 1
        index = torch.randint(0, 5, input_shape)
        op = ScatterOp(input_shape, dim, index, output_shape)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)

    def test_vforward_vbackward(self):
        torch.manual_seed(42)
        input_shape = torch.Size([4, 3, 6])
        output_shape = torch.Size([4, 5, 6])
        dim = 1
        index = torch.randint(0, 5, input_shape)
        op = ScatterOp(input_shape, dim, index, output_shape)
        x = torch.randn(input_shape)
        check_vforward_vbackward(op, x, (2, 3))


class TestGetIndicesOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([5, 6])
        output_shape = torch.Size([3, 4])
        indices = (
            torch.randint(0, 5, output_shape),
            torch.randint(0, 6, output_shape),
        )
        op = GetIndicesOp(indices, input_shape, output_shape)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


class TestSetIndicesOp:
    def test_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([3, 4])
        output_shape = torch.Size([5, 6])
        indices = (
            torch.randint(0, 5, input_shape),
            torch.randint(0, 6, input_shape),
        )
        op = SetIndicesOp(indices, input_shape, output_shape)

        x = torch.randn(input_shape)
        grad = torch.randn(op.output_shape)
        check_backward_vs_vjp(op, x, grad)


# ============================================================================
# Composition tests
# ============================================================================


class TestComposition:
    def test_reshape_then_permute_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([2, 3, 4])

        op1 = ReshapeOp(input_shape, (6, 4))
        op2 = PermuteOp(op1.output_shape, (1, 0))
        composed = op2 @ op1

        x = torch.randn(input_shape)
        grad = torch.randn(composed.output_shape)
        check_backward_vs_vjp(composed, x, grad)

    def test_narrow_then_expand_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([10, 1, 5])

        op1 = NarrowOp(input_shape, dim=0, start=2, length=5)
        op2 = ExpandOp(op1.output_shape, (5, 3, 5))
        composed = op2 @ op1

        x = torch.randn(input_shape)
        grad = torch.randn(composed.output_shape)
        check_backward_vs_vjp(composed, x, grad)

    def test_sum_of_ops_backward_vs_vjp(self):
        torch.manual_seed(42)
        input_shape = torch.Size([3, 4])

        op1 = FlipOp(input_shape, dims=(0,))
        op2 = FlipOp(input_shape, dims=(1,))
        summed = op1 + op2

        x = torch.randn(input_shape)
        grad = torch.randn(summed.output_shape)
        check_backward_vs_vjp(summed, x, grad)


# ============================================================================
# Stress tests with multiple random seeds
# ============================================================================


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_reshape_random_shapes(seed):
    torch.manual_seed(seed)
    # Generate random compatible shapes
    total = 24
    input_shape = torch.Size([2, 3, 4])
    target_shape = (4, 6)

    op = ReshapeOp(input_shape, target_shape)
    x = torch.randn(input_shape)
    grad = torch.randn(op.output_shape)
    check_backward_vs_vjp(op, x, grad)


@pytest.mark.parametrize("seed", [200, 201, 202])
def test_gather_scatter_roundtrip(seed):
    """Verify gather followed by scatter can reconstruct (for unique indices)."""
    torch.manual_seed(seed)
    shape = torch.Size([5, 6])
    dim = 1

    # Create unique indices along dim
    idx = torch.stack([torch.randperm(6)[:4] for _ in range(5)])
    gather_op = GatherOp(shape, dim, idx)
    scatter_op = ScatterOp(gather_op.output_shape, dim, idx, shape)

    x = torch.randn(shape)
    gathered = gather_op.forward(x)
    scattered = scatter_op.forward(gathered)

    # Only the selected elements should match
    for i in range(5):
        for j, k in enumerate(idx[i]):
            assert torch.allclose(scattered[i, k], x[i, k], atol=1e-6)


@pytest.mark.parametrize("seed", [300, 301, 302])
def test_expression_chain_soundness(seed):
    """Test a chain of shape ops maintains sound bounds."""
    torch.manual_seed(seed)
    center_val = torch.randn(4, 6)
    x_expr, center = _make_input(center_val)

    # Chain: reshape -> permute -> narrow -> expand
    y = x_expr.reshape(2, 3, 4)
    y = y.permute(2, 0, 1)
    y = y.narrow(0, 1, 2)
    y = y.unsqueeze(0)

    ub, lb = y.ublb()
    samples = _sample_inputs(center, n=1000)
    outputs = samples.reshape(-1, 2, 3, 4).permute(0, 3, 1, 2).narrow(1, 1, 2).unsqueeze(1)
    _check_bounds(outputs, ub, lb)
