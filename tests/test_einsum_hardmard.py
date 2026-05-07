from itertools import product

import torch

from boundlab.linearop._einsum import EinsumOp


def _expected_hardmard_jacobian(weight: torch.Tensor, n_input_dims: int) -> torch.Tensor:
    output_shape = tuple(weight.shape)
    input_shape = output_shape[-n_input_dims:] if n_input_dims else ()
    jacobian = torch.zeros(output_shape + input_shape, dtype=weight.dtype)

    for output_index in product(*[range(size) for size in output_shape]):
        input_index = output_index[-n_input_dims:] if n_input_dims else ()
        jacobian[output_index + input_index] = weight[output_index]
    return jacobian


def test_from_hardmard_default_matches_direct_elementwise_multiply():
    weight = torch.tensor(
        [
            [1.0, -2.0, 0.5],
            [3.0, -0.25, 4.0],
        ]
    )
    x = torch.tensor(
        [
            [-1.0, 2.0, 3.0],
            [0.5, -4.0, 1.5],
        ]
    )
    grad = torch.tensor(
        [
            [0.25, -1.0, 2.0],
            [3.0, 0.5, -0.75],
        ]
    )

    op = EinsumOp.from_hardmard(weight)

    assert op.input_shape == weight.shape
    assert op.output_shape == weight.shape
    assert torch.allclose(op.forward(x), weight * x)
    assert torch.allclose(op.backward(grad), weight * grad)
    assert torch.allclose(op.jacobian(), _expected_hardmard_jacobian(weight, 2))


def test_from_hardmard_trailing_input_dims_match_direct_broadcast_multiply():
    weight = torch.arange(24.0).reshape(2, 3, 4) / 10.0 - 1.0
    x = torch.tensor([1.5, -2.0, 0.25, 3.0])
    grad = torch.arange(24.0).reshape(2, 3, 4) / 7.0

    op = EinsumOp.from_hardmard(weight, n_input_dims=1)

    assert op.input_shape == torch.Size([4])
    assert op.output_shape == weight.shape
    assert torch.allclose(op.forward(x), weight * x.reshape(1, 1, 4))
    assert torch.allclose(op.backward(grad), (weight * grad).sum(dim=(0, 1)))
    assert torch.allclose(op.jacobian(), _expected_hardmard_jacobian(weight, 1))


def test_from_hardmard_vforward_vbackward_match_direct_elementwise_multiply():
    weight = torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [-3.0, 4.0, 1.25],
        ]
    )
    op = EinsumOp.from_hardmard(weight)
    x_batched = torch.arange(24.0).reshape(2, 3, 4) / 5.0 - 2.0
    grad_batched = torch.arange(30.0).reshape(5, 2, 3) / 6.0 - 1.0

    assert torch.allclose(op.vforward(x_batched), weight.unsqueeze(-1) * x_batched)
    assert torch.allclose(op.vbackward(grad_batched), weight.unsqueeze(0) * grad_batched)
