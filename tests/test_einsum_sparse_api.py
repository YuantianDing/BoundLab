"""Tests for the compact sparse-backed ``EinsumOp`` API."""

import torch
import boundlab.expr as expr
from torch.func import vjp

from boundlab.linearop import EinsumOp, ScalarOp
from boundlab.linearop._base import LinearOpFlags


def _make(shape, input_dims, output_dims, seed=0):
    torch.manual_seed(seed)
    return EinsumOp(torch.randn(shape), list(input_dims), list(output_dims))


def _assert_backward_matches_vjp(op, x, grad, atol=1e-5):
    _, vjp_fn = vjp(op.forward, x)
    (expected,) = vjp_fn(grad)
    assert torch.allclose(op.backward(grad), expected, atol=atol)


def _assert_jacobian_applies(op, x, atol=1e-5):
    jac = op.jacobian()
    assert jac.shape == (*op.output_shape, *op.input_shape)
    matrix = jac.reshape(op.output_shape.numel(), op.input_shape.numel())
    expected = matrix @ x.reshape(-1)
    assert torch.allclose(op.forward(x).reshape(-1), expected, atol=atol)


def _assert_vforward_matches_loop(op, x, batch_shape=(2, 3), atol=1e-5):
    xb = x.reshape(*x.shape, *([1] * len(batch_shape))).expand(*x.shape, *batch_shape)
    yb = op.vforward(xb)
    assert yb.shape == (*op.output_shape, *batch_shape)
    for index in torch.cartesian_prod(*[torch.arange(n) for n in batch_shape]):
        index = tuple(int(i) for i in index)
        assert torch.allclose(yb[(..., *index)], op.forward(xb[(..., *index)]), atol=atol)


def _assert_vbackward_matches_loop(op, grad, batch_shape=(2, 3), atol=1e-5):
    gb = grad.reshape(*([1] * len(batch_shape)), *grad.shape).expand(*batch_shape, *grad.shape)
    xb = op.vbackward(gb)
    assert xb.shape == (*batch_shape, *op.input_shape)
    for index in torch.cartesian_prod(*[torch.arange(n) for n in batch_shape]):
        index = tuple(int(i) for i in index)
        assert torch.allclose(xb[index], op.backward(gb[index]), atol=atol)


def test_full_einsum_matches_matrix_multiply():
    torch.manual_seed(0)
    weight = torch.randn(5, 3)
    op = EinsumOp.from_full(weight, input_dim=1)
    x = torch.randn(3)
    grad = torch.randn(5)

    assert op.input_shape == torch.Size([3])
    assert op.output_shape == torch.Size([5])
    assert torch.allclose(op.forward(x), weight @ x)
    _assert_backward_matches_vjp(op, x, grad)
    _assert_jacobian_applies(op, x)


def test_hadamard_einsum_matches_elementwise_multiply():
    torch.manual_seed(1)
    coeff = torch.randn(3, 4)
    op = EinsumOp.from_hardmard(coeff)
    x = torch.randn(3, 4)
    grad = torch.randn(3, 4)

    assert op.input_shape == torch.Size([3, 4])
    assert op.output_shape == torch.Size([3, 4])
    assert torch.allclose(op.forward(x), coeff * x)
    _assert_backward_matches_vjp(op, x, grad)
    _assert_jacobian_applies(op, x)


def test_mixed_einsum_dims_match_torch_einsum():
    torch.manual_seed(2)
    coeff = torch.randn(2, 3, 4)
    op = EinsumOp(coeff, input_dims=[1, 2], output_dims=[0, 1])
    x = torch.randn(3, 4)
    grad = torch.randn(2, 3)

    assert torch.allclose(op.forward(x), torch.einsum("abc,bc->ab", coeff, x))
    _assert_backward_matches_vjp(op, x, grad)
    _assert_jacobian_applies(op, x)


def test_batched_forward_and_backward_use_sparse_executor():
    op = _make((2, 3, 4), [1, 2], [0, 1], seed=3)
    _assert_vforward_matches_loop(op, torch.randn(3, 4))
    _assert_vbackward_matches_loop(op, torch.randn(2, 3))


def test_composition_preserves_semantics_without_fusion_api():
    a = _make((5, 4), [1], [0], seed=4)
    b = _make((4, 3), [1], [0], seed=5)
    composed = a @ b
    x = torch.randn(3)
    grad = torch.randn(5)

    assert torch.allclose(composed.forward(x), a.forward(b.forward(x)), atol=1e-5)
    assert torch.allclose(composed.backward(grad), b.backward(a.backward(grad)), atol=1e-5)


def test_addition_and_scaling_preserve_semantics():
    a = _make((5, 3), [1], [0], seed=6)
    b = _make((5, 3), [1], [0], seed=7)
    combined = 2.0 * a + b
    x = torch.randn(3)
    grad = torch.randn(5)

    assert torch.allclose(combined.forward(x), 2.0 * a.forward(x) + b.forward(x), atol=1e-5)
    assert torch.allclose(combined.backward(grad), 2.0 * a.backward(grad) + b.backward(grad), atol=1e-5)


def test_scalarop_composes_with_sparse_einsum():
    op = _make((3, 4), [1], [0], seed=8)
    scalar = ScalarOp(2.5, op.input_shape)
    composed = op @ scalar
    x = torch.randn(op.input_shape)

    assert torch.allclose(composed.forward(x), op.forward(2.5 * x), atol=1e-5)


def test_sum_and_norm_helpers_match_jacobian_rows():
    op = _make((4, 3), [1], [0], seed=9)
    jac = op.jacobian()
    input_rows = jac.reshape(op.output_shape.numel(), -1)
    output_rows = jac.permute(1, 0).reshape(op.input_shape.numel(), -1)

    assert torch.allclose(op.norm_input(2).jacobian(), input_rows.norm(2, dim=1).reshape(op.output_shape), atol=1e-5)
    assert torch.allclose(op.norm_output(2).jacobian(), output_rows.norm(2, dim=1).reshape(op.input_shape), atol=1e-5)
    assert op.sum_input().input_shape == torch.Size([])
    assert op.sum_output().output_shape == torch.Size([])


def test_norm_input_regression_mixed_sign_single_row():
    weight = torch.tensor([[1.0, -2.0]])
    x = expr.ConstVal(torch.zeros(2)) + expr.LpEpsilon([2])
    op0 = EinsumOp.from_full(weight, 1) @ ScalarOp(0.2, torch.Size([2]))
    print(op0)
    y = op0(x)
    assert LinearOpFlags.IS_NON_NEGATIVE not in op0.flags

    op = next(iter(y.children_dict.values()))
    assert torch.allclose(op.jacobian(), torch.tensor([[0.2, -0.4]]), atol=1e-6)
    assert torch.allclose(op.norm_input(1).jacobian(), torch.tensor([0.6]), atol=1e-6)

    ub, lb = y.ublb()
    assert torch.allclose(ub, torch.tensor([0.6]), atol=1e-6)
    assert torch.allclose(lb, torch.tensor([-0.6]), atol=1e-6)
