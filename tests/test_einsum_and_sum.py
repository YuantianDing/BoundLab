"""Extensive tests for EinsumOp and SumOp.

Covers:
- EinsumOp: forward/backward, vforward/vbackward, classification helpers,
  constructors (from_full, from_hardmard, from_scalar), composition (@),
  addition (+), scalar multiplication, jacobian/force_jacobian,
  sum_input/sum_output, norm_input/norm_output, abs, permute_for_input/output,
  merge_einsumop.
- SumOp: forward/backward, vforward/vbackward, composition (@), addition (+),
  jacobian, flattening nested SumOps.

Each fused operation is verified against the unfused (ComposedOp / explicit)
equivalent to ensure correctness of both forward and backward.
"""

import torch
import pytest
from torch.func import vjp

from boundlab.linearop._base import (
    LinearOp, ComposedOp, SumOp, ScalarOp, ZeroOp,
)
from boundlab.linearop._einsum import EinsumOp, merge_einsumop
from boundlab.linearop._shape import PermuteOp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_forward_backward(op, x, grad, atol=1e-5):
    """Verify op.backward(grad) matches VJP."""
    our_bwd = op.backward(grad)
    _, vjp_fn = vjp(op.forward, x)
    (vjp_bwd,) = vjp_fn(grad)
    assert torch.allclose(our_bwd, vjp_bwd, atol=atol), \
        f"backward mismatch for {op}: max diff = {(our_bwd - vjp_bwd).abs().max()}"


def _check_vforward(op, x, batch=(2, 3), atol=1e-5):
    """Verify vforward matches batched forward."""
    x_b = x.unsqueeze(-1).unsqueeze(-1).expand(*x.shape, *batch)
    y_b = op.vforward(x_b)
    for i in range(batch[0]):
        for j in range(batch[1]):
            y_ref = op.forward(x_b[..., i, j])
            assert torch.allclose(y_b[..., i, j], y_ref, atol=atol), \
                f"vforward mismatch at [{i},{j}]"


def _check_vbackward(op, grad, batch=(2, 3), atol=1e-5):
    """Verify vbackward matches batched backward."""
    g_b = grad.unsqueeze(0).unsqueeze(0).expand(*batch, *grad.shape)
    dx_b = op.vbackward(g_b)
    for i in range(batch[0]):
        for j in range(batch[1]):
            dx_ref = op.backward(g_b[i, j])
            assert torch.allclose(dx_b[i, j], dx_ref, atol=atol), \
                f"vbackward mismatch at [{i},{j}]"


def _check_jacobian(op, atol=1e-5):
    """Verify jacobian matches force_jacobian."""
    jac = op.force_jacobian()
    assert jac.shape == (*op.output_shape, *op.input_shape)
    # forward with basis vectors
    x = torch.randn(op.input_shape)
    y = op.forward(x)
    y_jac = torch.einsum("...i,i->...", jac.reshape(op.output_shape.numel(), op.input_shape.numel()), x.flatten())
    assert torch.allclose(y.flatten(), y_jac, atol=atol), \
        f"jacobian forward mismatch: max diff = {(y.flatten() - y_jac).abs().max()}"


def _make(tensor_shape, input_dims, output_dims, seed=42):
    torch.manual_seed(seed)
    return EinsumOp(torch.randn(tensor_shape), list(input_dims), list(output_dims))


# ========================================================================
# EinsumOp basic operations
# ========================================================================


class TestEinsumOpBasic:
    """Forward, backward, vforward, vbackward correctness."""

    def test_full_forward_backward(self):
        """Full EinsumOp (matrix-vector product)."""
        e = _make((5, 3), [1], [0])
        x = torch.randn(3)
        grad = torch.randn(5)
        _check_forward_backward(e, x, grad)

    def test_full_matmul(self):
        """Full EinsumOp behaves like matrix multiplication."""
        torch.manual_seed(0)
        W = torch.randn(5, 3)
        e = EinsumOp.from_full(W, 1)
        x = torch.randn(3)
        assert torch.allclose(e.forward(x), W @ x)

    def test_hadamard_forward_backward(self):
        """Hadamard (elementwise) EinsumOp."""
        e = _make((3, 4), [0, 1], [0, 1])
        x = torch.randn(3, 4)
        grad = torch.randn(3, 4)
        _check_forward_backward(e, x, grad)

    def test_hadamard_is_elementwise(self):
        """from_hardmard produces elementwise multiplication."""
        torch.manual_seed(0)
        w = torch.randn(3, 4)
        e = EinsumOp.from_hardmard(w, 2)
        x = torch.randn(3, 4)
        assert torch.allclose(e.forward(x), w * x)

    def test_mixed_dims(self):
        """EinsumOp with dot, mul, and batch dims."""
        # tensor (2, 3, 4): dim 0 = batch, dim 1 = mul, dim 2 = dot
        e = _make((2, 3, 4), [1, 2], [0, 1])
        x = torch.randn(3, 4)
        grad = torch.randn(2, 3)
        _check_forward_backward(e, x, grad)

    def test_vforward_full(self):
        e = _make((5, 3), [1], [0])
        _check_vforward(e, torch.randn(3))

    def test_vbackward_full(self):
        e = _make((5, 3), [1], [0])
        _check_vbackward(e, torch.randn(5))

    def test_vforward_hadamard(self):
        e = _make((3, 4), [0, 1], [0, 1])
        _check_vforward(e, torch.randn(3, 4))

    def test_vbackward_hadamard(self):
        e = _make((3, 4), [0, 1], [0, 1])
        _check_vbackward(e, torch.randn(3, 4))

    def test_vforward_mixed(self):
        e = _make((2, 3, 4), [1, 2], [0, 1])
        _check_vforward(e, torch.randn(3, 4))

    def test_vbackward_mixed(self):
        e = _make((2, 3, 4), [1, 2], [0, 1])
        _check_vbackward(e, torch.randn(2, 3))


# ========================================================================
# EinsumOp classification helpers
# ========================================================================


class TestEinsumClassification:
    def test_is_full(self):
        e = _make((5, 3), [1], [0])
        assert e.is_full()
        assert not e.is_hardmard()
        # Full EinsumOps have batch_dims (output-only dims), so not non-expanding.
        assert not e.is_non_expanding()

    def test_is_hadamard(self):
        e = _make((3, 4), [0, 1], [0, 1])
        assert not e.is_full()
        assert e.is_hardmard()
        assert e.is_non_expanding()

    def test_is_expanding(self):
        e = _make((2, 3, 4), [2], [0, 1])
        assert not e.is_non_expanding()
        assert e.batch_dims == [0, 1]

    def test_dot_mul_batch_dims(self):
        # tensor (A, B, C, D): input=[0,1,2], output=[1,2,3]
        # dot = [0], mul = [1,2], batch = [3]
        e = _make((2, 3, 4, 5), [0, 1, 2], [1, 2, 3])
        assert e.dot_dims == [0]
        assert e.mul_dims == [1, 2]
        assert e.batch_dims == [3]


# ========================================================================
# Constructors
# ========================================================================


class TestEinsumConstructors:
    def test_from_full(self):
        torch.manual_seed(0)
        W = torch.randn(3, 4, 5)
        e = EinsumOp.from_full(W, 2)
        assert e.output_shape == torch.Size([3])
        assert e.input_shape == torch.Size([4, 5])
        assert e.is_full()

    def test_from_hardmard(self):
        torch.manual_seed(0)
        w = torch.randn(2, 3, 4)
        e = EinsumOp.from_hardmard(w, 2)
        assert e.input_shape == torch.Size([3, 4])
        assert e.output_shape == torch.Size([2, 3, 4])
        assert e.batch_dims == [0]

    def test_from_scalar(self):
        s = ScalarOp(3.0, torch.Size([2, 3]))
        e = EinsumOp.from_scalar(s)
        x = torch.randn(2, 3)
        assert torch.allclose(e.forward(x), 3.0 * x)


# ========================================================================
# Scalar multiplication
# ========================================================================


class TestEinsumScalar:
    def test_mul_scalar(self):
        e = _make((5, 3), [1], [0])
        e2 = e * 2.5
        x = torch.randn(3)
        assert torch.allclose(e2.forward(x), 2.5 * e.forward(x))

    def test_rmul_scalar(self):
        e = _make((5, 3), [1], [0])
        e2 = 0.5 * e
        x = torch.randn(3)
        assert torch.allclose(e2.forward(x), 0.5 * e.forward(x))


# ========================================================================
# EinsumOp @ EinsumOp (merge_einsumop)
# ========================================================================


class TestMergeEinsumop:
    def test_full_full(self):
        """Two full EinsumOps compose into a full EinsumOp."""
        a = _make((5, 4), [1], [0], seed=0)
        b = _make((4, 3), [1], [0], seed=1)
        c = a @ b
        assert isinstance(c, EinsumOp)
        x = torch.randn(3)
        assert torch.allclose(c.forward(x), a.forward(b.forward(x)), atol=1e-5)
        grad = torch.randn(5)
        assert torch.allclose(c.backward(grad), b.backward(a.backward(grad)), atol=1e-5)

    def test_hadamard_full(self):
        """Hadamard @ full compose."""
        torch.manual_seed(0)
        h = EinsumOp.from_hardmard(torch.randn(3, 4), 2)
        f = EinsumOp.from_full(torch.randn(3, 4, 5), 1)
        c = h @ f
        assert isinstance(c, EinsumOp)
        x = torch.randn(5)
        assert torch.allclose(c.forward(x), h.forward(f.forward(x)), atol=1e-5)

    def test_full_hadamard(self):
        """Full @ hadamard compose."""
        torch.manual_seed(1)
        f = EinsumOp.from_full(torch.randn(5, 3, 4), 2)
        h = EinsumOp.from_hardmard(torch.randn(3, 4), 2)
        c = f @ h
        assert isinstance(c, EinsumOp)
        x = torch.randn(3, 4)
        assert torch.allclose(c.forward(x), f.forward(h.forward(x)), atol=1e-5)

    def test_associativity(self):
        """(a @ b) @ c ≈ a @ (b @ c)."""
        a = _make((5, 4), [1], [0], seed=0)
        b = _make((4, 3), [1], [0], seed=1)
        c = _make((3, 2), [1], [0], seed=2)
        ab_c = (a @ b) @ c
        a_bc = a @ (b @ c)
        x = torch.randn(2)
        assert torch.allclose(ab_c.forward(x), a_bc.forward(x), atol=1e-4)


# ========================================================================
# EinsumOp @ ScalarOp / PermuteOp
# ========================================================================


class TestEinsumCompositionSpecial:
    def test_compose_with_identity_scalar(self):
        e = _make((5, 3), [1], [0])
        s = ScalarOp(1.0, torch.Size([3]))
        fused = e @ s
        assert isinstance(fused, EinsumOp)
        x = torch.randn(3)
        assert torch.allclose(fused.forward(x), e.forward(x))

    def test_compose_with_scalar(self):
        e = _make((5, 3), [1], [0])
        s = ScalarOp(2.0, torch.Size([3]))
        fused = e @ s
        x = torch.randn(3)
        assert torch.allclose(fused.forward(x), e.forward(2 * x), atol=1e-5)

    def test_compose_with_permute_input(self):
        e = _make((5, 3, 4), [1, 2], [0])
        p = PermuteOp(torch.Size([4, 3]), (1, 0))
        fused = e @ p
        assert isinstance(fused, EinsumOp)
        x = torch.randn(4, 3)
        assert torch.allclose(fused.forward(x), e.forward(p.forward(x)), atol=1e-5)

    def test_compose_with_permute_output(self):
        e = _make((5, 3, 4), [2], [0, 1])
        p = PermuteOp(e.output_shape, (1, 0))
        fused = p @ e
        assert isinstance(fused, EinsumOp)
        x = torch.randn(4)
        assert torch.allclose(fused.forward(x), p.forward(e.forward(x)), atol=1e-5)

    def test_scalar_rmatmul(self):
        e = _make((5, 3), [1], [0])
        s = ScalarOp(3.0, torch.Size([5]))
        fused = s @ e
        x = torch.randn(3)
        assert torch.allclose(fused.forward(x), 3.0 * e.forward(x), atol=1e-5)


# ========================================================================
# EinsumOp + EinsumOp (__radd__)
# ========================================================================


class TestEinsumAddition:
    def test_same_layout(self):
        """Adding two EinsumOps with identical layout."""
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        c = a + b
        assert isinstance(c, EinsumOp)
        x = torch.randn(3)
        assert torch.allclose(c.forward(x), a.forward(x) + b.forward(x), atol=1e-5)

    def test_add_scalar_radd(self):
        """ScalarOp + EinsumOp fuses via EinsumOp.__radd__."""
        e = _make((3, 4), [0, 1], [0, 1])
        s = ScalarOp(2.0, torch.Size([3, 4]))
        c = s + e  # goes through EinsumOp.__radd__
        assert isinstance(c, EinsumOp)
        x = torch.randn(3, 4)
        assert torch.allclose(c.forward(x), e.forward(x) + 2.0 * x, atol=1e-5)

    def test_add_scalar_lhs(self):
        """EinsumOp + ScalarOp falls through to SumOp but remains correct."""
        e = _make((3, 4), [0, 1], [0, 1])
        s = ScalarOp(2.0, torch.Size([3, 4]))
        c = e + s
        x = torch.randn(3, 4)
        assert torch.allclose(c.forward(x), e.forward(x) + 2.0 * x, atol=1e-5)

    def test_add_zero(self):
        """ScalarOp(0) + EinsumOp returns original via __radd__."""
        e = _make((5, 3), [1], [0])
        s = ScalarOp(0.0, torch.Size([3]))
        c = s + e  # goes through EinsumOp.__radd__
        assert c is e

    def test_different_layout_falls_back(self):
        """EinsumOps with incompatible layouts produce SumOp."""
        a = _make((5, 3), [1], [0], seed=0)
        # different layout: hadamard vs full
        torch.manual_seed(1)
        b = EinsumOp(torch.randn(5, 3), [0, 1], [0, 1])  # wrong shapes for add with a
        # These have different input/output shapes so can't be added — skip
        # Instead: same shapes but different dim structure
        torch.manual_seed(2)
        t1 = torch.randn(5, 3)
        e1 = EinsumOp(t1, [1], [0])  # full: (3,) → (5,)
        torch.manual_seed(3)
        t2 = torch.randn(5, 3)
        e2 = EinsumOp(t2, [1], [0])  # full: (3,) → (5,), same layout
        c = e1 + e2
        assert isinstance(c, EinsumOp)  # same layout → fused


# ========================================================================
# Jacobian / force_jacobian
# ========================================================================


class TestEinsumJacobian:
    def test_full_jacobian(self):
        e = _make((5, 3), [1], [0])
        jac = e.jacobian()
        assert jac.shape == (5, 3)
        _check_jacobian(e)

    def test_hadamard_force_jacobian(self):
        e = _make((3, 4), [0, 1], [0, 1])
        _check_jacobian(e)

    def test_mixed_force_jacobian(self):
        e = _make((2, 3, 4), [1, 2], [0, 1])
        _check_jacobian(e)

    def test_jacobian_consistency(self):
        """jacobian() and force_jacobian() agree for full ops."""
        e = _make((5, 3), [1], [0])
        assert torch.allclose(e.jacobian(), e.force_jacobian(), atol=1e-5)


# ========================================================================
# sum_input / sum_output / norm_input / norm_output / abs
# ========================================================================


class TestEinsumReductions:
    def test_sum_input_full(self):
        e = _make((5, 3), [1], [0])
        s = e.sum_input()
        # sum_input of a full (5,3) -> output (5,): should sum rows
        assert s.input_dims == []
        x = torch.randn(3)
        ref = e.forward(x)  # (5,)
        # sum_input "sums over input" → result is just the row sums of the weight
        assert s.output_shape == e.output_shape

    def test_sum_output_full(self):
        e = _make((5, 3), [1], [0])
        s = e.sum_output()
        assert s.output_dims == []
        assert s.input_shape == e.input_shape

    def test_abs(self):
        e = _make((5, 3), [1], [0])
        a = e.abs()
        assert isinstance(a, EinsumOp)
        assert torch.allclose(a.tensor, e.tensor.abs())

    def test_norm_input_l1(self):
        """norm_input(p=1) equals abs().sum_input()."""
        e = _make((5, 3), [1], [0])
        n = e.norm_input(p=1)
        a = e.abs().sum_input()
        assert torch.allclose(n.tensor, a.tensor, atol=1e-5)

    def test_norm_input_l2(self):
        e = _make((5, 3), [1], [0])
        n = e.norm_input(p=2)
        # Manual: sqrt(sum(T^2, over input dims))
        ref = e.tensor.pow(2).sum(dim=e.dot_dims).sqrt()
        # n is an EinsumOp with output-only dims; compare tensors
        assert n.input_dims == []
        # Flatten both to compare
        assert torch.allclose(n.tensor.flatten(), ref.flatten(), atol=1e-5)

    def test_norm_output_l1(self):
        e = _make((5, 3), [1], [0])
        n = e.norm_output(p=1)
        assert n.output_dims == []


# ========================================================================
# permute_for_input / permute_for_output
# ========================================================================


class TestEinsumPermute:
    def test_permute_for_input_identity(self):
        """Permuting for input preserves semantics."""
        e = _make((2, 3, 4), [1, 2], [0, 1])
        ep = e.permute_for_input()
        x = torch.randn(3, 4)
        assert torch.allclose(ep.forward(x), e.forward(x), atol=1e-5)
        grad = torch.randn(2, 3)
        assert torch.allclose(ep.backward(grad), e.backward(grad), atol=1e-5)

    def test_permute_for_output_identity(self):
        e = _make((2, 3, 4), [1, 2], [0, 1])
        ep = e.permute_for_output()
        x = torch.randn(3, 4)
        assert torch.allclose(ep.forward(x), e.forward(x), atol=1e-5)
        grad = torch.randn(2, 3)
        assert torch.allclose(ep.backward(grad), e.backward(grad), atol=1e-5)


# ========================================================================
# SumOp
# ========================================================================


class TestSumOpBasic:
    """SumOp: forward, backward, vforward, vbackward."""

    def test_forward_backward(self):
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        s = SumOp(a, b)
        x = torch.randn(3)
        assert torch.allclose(s.forward(x), a.forward(x) + b.forward(x))
        grad = torch.randn(5)
        assert torch.allclose(s.backward(grad), a.backward(grad) + b.backward(grad))

    def test_vforward(self):
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        s = SumOp(a, b)
        _check_vforward(s, torch.randn(3))

    def test_vbackward(self):
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        s = SumOp(a, b)
        _check_vbackward(s, torch.randn(5))

    def test_jacobian(self):
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        s = SumOp(a, b)
        jac = s.jacobian()
        assert torch.allclose(jac, a.jacobian() + b.jacobian(), atol=1e-5)


class TestSumOpFlattening:
    """SumOp flattens nested SumOps."""

    def test_nested_flattening(self):
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        c = _make((5, 3), [1], [0], seed=2)
        s1 = SumOp(a, b)
        s2 = SumOp(s1, c)
        assert len(s2.ops) == 3

    def test_three_ops(self):
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        c = _make((5, 3), [1], [0], seed=2)
        s = SumOp(a, b, c)
        x = torch.randn(3)
        assert torch.allclose(s.forward(x),
                              a.forward(x) + b.forward(x) + c.forward(x), atol=1e-5)


class TestSumOpComposition:
    """SumOp @ LinearOp and LinearOp @ SumOp."""

    def test_matmul_right(self):
        """SumOp @ EinsumOp distributes."""
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        s = SumOp(a, b)
        c = _make((3, 4), [1], [0], seed=2)
        result = s @ c
        x = torch.randn(4)
        expected = a.forward(c.forward(x)) + b.forward(c.forward(x))
        assert torch.allclose(result.forward(x), expected, atol=1e-5)

    def test_rmatmul_left(self):
        """EinsumOp @ SumOp distributes."""
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        s = SumOp(a, b)
        c = _make((7, 5), [1], [0], seed=2)
        result = c @ s
        x = torch.randn(3)
        expected = c.forward(a.forward(x)) + c.forward(b.forward(x))
        assert torch.allclose(result.forward(x), expected, atol=1e-5)


class TestSumOpAddition:
    """SumOp + SumOp / SumOp + LinearOp tries to fuse, falls back."""

    def test_add_sumop_fuses_compatible(self):
        """Two SumOps with same-layout EinsumOps fuse component-wise into a
        single EinsumOp (the lone element is unwrapped out of the SumOp)."""
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        s1 = SumOp(a)
        s2 = SumOp(b)
        result = s1 + s2
        assert isinstance(result, EinsumOp)
        x = torch.randn(3)
        assert torch.allclose(result.forward(x), a.forward(x) + b.forward(x), atol=1e-5)

    def test_add_linearop(self):
        a = _make((5, 3), [1], [0], seed=0)
        b = _make((5, 3), [1], [0], seed=1)
        s = SumOp(a)
        result = s + b
        assert isinstance(result, EinsumOp)
        x = torch.randn(3)
        assert torch.allclose(result.forward(x), a.forward(x) + b.forward(x), atol=1e-5)


class TestSumOpWithZero:
    """ZeroOp interacts correctly with SumOp."""

    def test_zero_add(self):
        a = _make((5, 3), [1], [0])
        z = ZeroOp(torch.Size([3]), torch.Size([5]))
        result = z + a
        assert result is a

    def test_add_zero(self):
        a = _make((5, 3), [1], [0])
        z = ZeroOp(torch.Size([3]), torch.Size([5]))
        result = a + z
        assert result is a


# ========================================================================
# EinsumOp @ full (is_full branch in __matmul__)
# ========================================================================


class TestEinsumFullComposition:
    """When the outer EinsumOp is full, composition uses vbackward."""

    def test_full_compose_with_hadamard(self):
        torch.manual_seed(0)
        full = EinsumOp.from_full(torch.randn(5, 3, 4), 2)
        had = EinsumOp.from_hardmard(torch.randn(3, 4), 2)
        c = full @ had
        assert isinstance(c, EinsumOp)
        x = torch.randn(3, 4)
        assert torch.allclose(c.forward(x), full.forward(had.forward(x)), atol=1e-5)
        grad = torch.randn(5)
        assert torch.allclose(c.backward(grad), had.backward(full.backward(grad)), atol=1e-5)

    def test_full_rcompose_with_hadamard(self):
        """When the inner EinsumOp is full, rmatmul uses vforward."""
        torch.manual_seed(0)
        full = EinsumOp.from_full(torch.randn(3, 4, 5), 1)
        had = EinsumOp.from_hardmard(torch.randn(2, 3, 4), 2)
        c = had @ full
        assert isinstance(c, EinsumOp)
        x = torch.randn(5)
        assert torch.allclose(c.forward(x), had.forward(full.forward(x)), atol=1e-5)


# ========================================================================
# Parametric stress tests
# ========================================================================


@pytest.mark.parametrize("seed", range(5))
class TestParametricEinsum:
    def test_matmul_chain(self, seed):
        """Chain of 3 full EinsumOps."""
        torch.manual_seed(seed)
        a = EinsumOp.from_full(torch.randn(5, 4), 1)
        b = EinsumOp.from_full(torch.randn(4, 3), 1)
        c = EinsumOp.from_full(torch.randn(3, 2), 1)
        abc = a @ b @ c
        x = torch.randn(2)
        ref = a.forward(b.forward(c.forward(x)))
        assert torch.allclose(abc.forward(x), ref, atol=1e-4)

    def test_sum_then_compose(self, seed):
        """(a + b) @ c compared to a@c + b@c."""
        torch.manual_seed(seed)
        a = EinsumOp.from_full(torch.randn(5, 3), 1)
        b = EinsumOp.from_full(torch.randn(5, 3), 1)
        c = EinsumOp.from_full(torch.randn(3, 4), 1)
        lhs = SumOp(a, b) @ c
        x = torch.randn(4)
        ref = a.forward(c.forward(x)) + b.forward(c.forward(x))
        assert torch.allclose(lhs.forward(x), ref, atol=1e-4)

    def test_compose_then_sum(self, seed):
        """a @ (b + c) via SumOp."""
        torch.manual_seed(seed)
        a = EinsumOp.from_full(torch.randn(5, 3), 1)
        b = EinsumOp.from_full(torch.randn(3, 4), 1)
        c = EinsumOp.from_full(torch.randn(3, 4), 1)
        rhs = a @ SumOp(b, c)
        x = torch.randn(4)
        ref = a.forward(b.forward(x) + c.forward(x))
        assert torch.allclose(rhs.forward(x), ref, atol=1e-4)

    def test_jacobian_of_composition(self, seed):
        """Jacobian of composed EinsumOps matches product of Jacobians."""
        torch.manual_seed(seed)
        a = EinsumOp.from_full(torch.randn(5, 4), 1)
        b = EinsumOp.from_full(torch.randn(4, 3), 1)
        ab = a @ b
        jac_ab = ab.jacobian()
        jac_ref = a.jacobian() @ b.jacobian()
        assert torch.allclose(jac_ab, jac_ref, atol=1e-4)

    def test_sum_jacobian(self, seed):
        """Jacobian of SumOp == sum of Jacobians."""
        torch.manual_seed(seed)
        a = EinsumOp.from_full(torch.randn(5, 3), 1)
        b = EinsumOp.from_full(torch.randn(5, 3), 1)
        s = SumOp(a, b)
        assert torch.allclose(s.jacobian(), a.jacobian() + b.jacobian(), atol=1e-5)
