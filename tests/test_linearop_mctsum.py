import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import Dense, TN


def _linearop_from_terms(
    output_size: int,
    input_size: int,
    term_specs: list[tuple[list[tuple[int, int]], list[float]]],
) -> tuple[LinearOp, torch.Tensor]:
    input_dim = Dim(input_size, 1.0, "lin_in")
    output_dim = Dim(output_size, 0.0, "lin_out")
    expected = torch.zeros(output_size, input_size)
    terms = []

    for term_idx, (pairs, values) in enumerate(term_specs):
        assert len(pairs) == len(values)
        edge_dim = Dim(len(pairs), 0.5 + term_idx / 100.0, f"edge{term_idx}")
        rows = torch.tensor([row for row, _ in pairs], dtype=torch.long)
        cols = torch.tensor([col for _, col in pairs], dtype=torch.long)
        vals = torch.tensor(values)
        expected[rows, cols] += vals

        table = TorchTable(
            columns=[output_dim, input_dim],
            data=[rows, cols],
            length=len(pairs),
        )
        terms.append(
            MultiCOOTensor(
                tn=TN.from_dense(Dense(vals, [edge_dim])),
                sparsify=MultiCOOSparsify([COOSparsify(edge_dim, table)]),
            )
        )

    return LinearOp(MultiCOOTensorSum(terms), [input_dim], [output_dim]), expected


def _linearop_from_matrix(matrix: torch.Tensor) -> LinearOp:
    pairs = [
        (row, col)
        for row in range(matrix.shape[0])
        for col in range(matrix.shape[1])
    ]
    values = [float(matrix[row, col]) for row, col in pairs]
    op, _ = _linearop_from_terms(matrix.shape[0], matrix.shape[1], [(pairs, values)])
    return op


def _assert_linearop_matches_matrix(op: LinearOp, matrix: torch.Tensor) -> None:
    x = torch.linspace(-0.75, 1.25, steps=matrix.shape[1])
    grad = torch.linspace(1.5, -0.5, steps=matrix.shape[0])

    assert torch.allclose(op.forward(x), matrix @ x)
    assert torch.allclose(op.backward(grad), matrix.T @ grad)
    assert torch.allclose(op.jacobian(), matrix)


def test_linearop_add_uses_multicootensorsum_and_matches_dense_matrix_sum():
    left_matrix = torch.tensor(
        [
            [1.0, -2.0, 0.5],
            [0.0, 3.0, -1.0],
        ]
    )
    right_matrix = torch.tensor(
        [
            [-4.0, 1.5, 2.0],
            [2.5, -0.5, 1.0],
        ]
    )

    result = _linearop_from_matrix(left_matrix) + _linearop_from_matrix(right_matrix)

    assert isinstance(result.tensor, MultiCOOTensorSum)
    _assert_linearop_matches_matrix(result, left_matrix + right_matrix)


def test_linearop_scalar_multiply_keeps_multicootensorsum_semantics():
    matrix = torch.tensor(
        [
            [2.0, -1.0, 0.0],
            [3.5, 0.5, -4.0],
        ]
    )
    op = _linearop_from_matrix(matrix)

    left_scaled = -2.5 * op
    right_scaled = op * -2.5

    assert isinstance(left_scaled.tensor, MultiCOOTensorSum)
    assert isinstance(right_scaled.tensor, MultiCOOTensorSum)
    _assert_linearop_matches_matrix(left_scaled, -2.5 * matrix)
    _assert_linearop_matches_matrix(right_scaled, -2.5 * matrix)


def test_linearop_abs_applies_multiplicative_to_multiterm_mctsum():
    term_specs = [
        ([(0, 0), (0, 1), (1, 1), (2, 3), (3, 0)], [1.0, -2.0, 0.5, 3.0, -1.5]),
        ([(0, 0), (1, 1), (1, 2), (2, 3), (3, 1)], [-2.0, 1.0, 4.0, -1.0, 0.5]),
        ([(0, 1), (1, 1), (2, 0), (2, 3), (3, 2)], [3.0, -5.0, 1.5, 2.0, -0.5]),
        ([(0, 2), (1, 1), (2, 0), (2, 3), (3, 3)], [-1.0, 2.0, -2.5, 1.0, 4.0]),
        ([(0, 0), (0, 3), (1, 1), (2, 1), (2, 3)], [0.25, -3.0, 1.5, -1.0, 2.5]),
        ([(0, 1), (1, 0), (1, 1), (2, 3), (3, 0)], [-1.0, 2.0, -0.5, -4.0, 3.0]),
    ]
    op, matrix = _linearop_from_terms(4, 4, term_specs)

    abs_op = op.abs()
    squared_op = op.abs(p=2)

    assert len(op.tensor.terms) == 6
    assert isinstance(abs_op.tensor, MultiCOOTensorSum)
    assert abs_op.flags & LinearOpFlags.IS_NON_NEGATIVE
    _assert_linearop_matches_matrix(abs_op, matrix.abs())
    _assert_linearop_matches_matrix(squared_op, matrix.pow(2))


def test_linearop_norm_input_matches_dense_row_norms_for_multiterm_mctsum():
    term_specs = [
        ([(0, 0), (0, 1), (1, 1), (2, 3), (3, 0)], [1.0, -2.0, 0.5, 3.0, -1.5]),
        ([(0, 0), (1, 1), (1, 2), (2, 3), (3, 1)], [-2.0, 1.0, 4.0, -1.0, 0.5]),
        ([(0, 1), (1, 1), (2, 0), (2, 3), (3, 2)], [3.0, -5.0, 1.5, 2.0, -0.5]),
        ([(0, 2), (1, 1), (2, 0), (2, 3), (3, 3)], [-1.0, 2.0, -2.5, 1.0, 4.0]),
        ([(0, 0), (0, 3), (1, 1), (2, 1), (2, 3)], [0.25, -3.0, 1.5, -1.0, 2.5]),
        ([(0, 1), (1, 0), (1, 1), (2, 3), (3, 0)], [-1.0, 2.0, -0.5, -4.0, 3.0]),
    ]
    op, matrix = _linearop_from_terms(4, 4, term_specs)

    l1 = op.norm_input(p=1)
    l2 = op.norm_input(p=2)

    assert l1.input_shape == torch.Size([])
    assert l1.output_shape == torch.Size([4])
    assert isinstance(l1.tensor, MultiCOOTensorSum)
    assert l1.flags & LinearOpFlags.IS_NON_NEGATIVE
    assert torch.allclose(l1.jacobian(), matrix.abs().sum(dim=1))
    assert torch.allclose(l2.jacobian(), matrix.pow(2).sum(dim=1).sqrt())
    assert torch.allclose(l1.forward(torch.tensor(1.0)), matrix.abs().sum(dim=1))
