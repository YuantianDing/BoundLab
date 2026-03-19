"""Tests for EinsumOp.jacobian_scatter correctness.

jacobian_scatter(src) should return src + self.force_jacobian() for any
EinsumOp configuration (full, hadamard, mixed).
"""

import torch
import pytest
from boundlab.linearop._einsum import EinsumOp


def check_jacobian_scatter(op: EinsumOp, src: torch.Tensor, atol: float = 1e-5):
    """Verify jacobian_scatter matches force_jacobian + src."""
    expected = op.force_jacobian() + src
    result = op.jacobian_scatter(src)
    assert result.shape == expected.shape, (
        f"Shape mismatch: got {result.shape}, expected {expected.shape}"
    )
    assert torch.allclose(result, expected, atol=atol), (
        f"jacobian_scatter mismatch for {op}:\n"
        f"  max diff: {(result - expected).abs().max():.6e}\n"
        f"  result sample: {result.flatten()[:5]}\n"
        f"  expected sample: {expected.flatten()[:5]}"
    )


class TestJacobianScatterFull:
    """Full EinsumOps (no mul_dims) — the simple path."""

    def test_matrix_vector(self):
        """from_full: tensor (3, 4), input_dim=1 => matrix-vector multiply."""
        torch.manual_seed(0)
        W = torch.randn(3, 4)
        op = EinsumOp.from_full(W, input_dim=1)
        src = torch.randn(op.output_shape + op.input_shape)
        check_jacobian_scatter(op, src)

    def test_batch_matmul(self):
        """from_full: tensor (2, 3, 4), input_dim=2 => batch output, 2-d input."""
        torch.manual_seed(1)
        W = torch.randn(2, 3, 4)
        op = EinsumOp.from_full(W, input_dim=2)
        src = torch.randn(op.output_shape + op.input_shape)
        check_jacobian_scatter(op, src)


class TestJacobianScatterHadamard:
    """Hadamard EinsumOps (no dot_dims, only mul_dims)."""

    def test_elementwise_1d(self):
        """from_hardmard: 1-d elementwise multiply."""
        torch.manual_seed(10)
        W = torch.randn(4)
        op = EinsumOp.from_hardmard(W, n_input_dims=1)
        src = torch.randn(op.output_shape + op.input_shape)
        check_jacobian_scatter(op, src)

    def test_elementwise_2d(self):
        """from_hardmard: 2-d elementwise multiply."""
        torch.manual_seed(11)
        W = torch.randn(3, 4)
        op = EinsumOp.from_hardmard(W, n_input_dims=2)
        src = torch.randn(op.output_shape + op.input_shape)
        check_jacobian_scatter(op, src)

    def test_broadcast_hadamard(self):
        """from_hardmard: tensor (2, 3, 4) with n_input_dims=2 => batch + elementwise."""
        torch.manual_seed(12)
        W = torch.randn(2, 3, 4)
        op = EinsumOp.from_hardmard(W, n_input_dims=2)
        # output_shape = (2, 3, 4), input_shape = (3, 4)
        src = torch.randn(op.output_shape + op.input_shape)
        check_jacobian_scatter(op, src)


class TestJacobianScatterMixed:
    """Mixed EinsumOps with both mul_dims and dot_dims."""

    def test_mixed_contract_and_elementwise(self):
        """tensor (3, 4, 5): input_dims=[1, 2], output_dims=[0, 1]
        mul_dims=[1], dot_dims=[2], batch_dims=[0]."""
        torch.manual_seed(20)
        W = torch.randn(3, 4, 5)
        op = EinsumOp(W, input_dims=[1, 2], output_dims=[0, 1])
        src = torch.randn(op.output_shape + op.input_shape)
        check_jacobian_scatter(op, src)

    def test_mixed_2(self):
        """tensor (2, 3, 4): input_dims=[0, 2], output_dims=[0, 1]
        mul_dims=[0], dot_dims=[2], batch_dims=[1]."""
        torch.manual_seed(21)
        W = torch.randn(2, 3, 4)
        op = EinsumOp(W, input_dims=[0, 2], output_dims=[0, 1])
        src = torch.randn(op.output_shape + op.input_shape)
        check_jacobian_scatter(op, src)

    def test_mixed_multidim_mul(self):
        """tensor (2, 3, 4, 5): input_dims=[1, 2, 3], output_dims=[0, 1, 2]
        mul_dims=[1, 2], dot_dims=[3], batch_dims=[0]."""
        torch.manual_seed(22)
        W = torch.randn(2, 3, 4, 5)
        op = EinsumOp(W, input_dims=[1, 2, 3], output_dims=[0, 1, 2])
        src = torch.randn(op.output_shape + op.input_shape)
        check_jacobian_scatter(op, src)


class TestJacobianScatterWithZeroSrc:
    """jacobian_scatter with zero src should just return the Jacobian."""

    def test_full_zero_src(self):
        torch.manual_seed(30)
        W = torch.randn(3, 4)
        op = EinsumOp.from_full(W, input_dim=1)
        src = torch.zeros(op.output_shape + op.input_shape)
        result = op.jacobian_scatter(src)
        expected = op.force_jacobian()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_hadamard_zero_src(self):
        torch.manual_seed(31)
        W = torch.randn(3, 4)
        op = EinsumOp.from_hardmard(W, n_input_dims=2)
        src = torch.zeros(op.output_shape + op.input_shape)
        result = op.jacobian_scatter(src)
        expected = op.force_jacobian()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_mixed_zero_src(self):
        torch.manual_seed(32)
        W = torch.randn(3, 4, 5)
        op = EinsumOp(W, input_dims=[1, 2], output_dims=[0, 1])
        src = torch.zeros(op.output_shape + op.input_shape)
        result = op.jacobian_scatter(src)
        expected = op.force_jacobian()
        assert torch.allclose(result, expected, atol=1e-5)
