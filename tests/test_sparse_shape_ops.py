import torch

from boundlab.linearop._expand import ExpandOp
from boundlab.linearop._indexing import GetIndicesOp, SetIndicesOp
from boundlab.linearop._indices import GatherOp, ScatterOp
from boundlab.linearop._permute import PermuteOp, TransposeOp
from boundlab.linearop._reshape import FlattenOp, ReshapeOp, SqueezeOp, UnflattenOp, UnsqueezeOp
from boundlab.linearop._shape import DiagOp, FlipOp, RepeatOp, RollOp, TileOp
from boundlab.linearop._slicing import GetSliceOp, SetSliceOp
from boundlab.sparse.coo import MultiCOOTensorSum


def _expected_jacobian(input_shape: torch.Size, output_shape: torch.Size, fn):
    columns = []
    for flat_idx in range(input_shape.numel()):
        basis = torch.zeros(input_shape)
        basis.reshape(-1)[flat_idx] = 1.0
        columns.append(fn(basis).reshape(-1))
    return torch.stack(columns, dim=1).reshape(output_shape + input_shape)


def test_reshapeop_constructs_sparse_matrix():
    op = ReshapeOp(torch.Size([2, 3]), (3, 2))
    x = torch.arange(6.0).reshape(2, 3)
    grad = torch.arange(6.0).reshape(3, 2) - 1.0
    expected_jacobian = _expected_jacobian(op.input_shape, op.output_shape, lambda t: t.reshape(3, 2))

    assert isinstance(op.tensor, MultiCOOTensorSum)
    assert len(op.tensor.terms) == 1
    assert len(op.tensor.terms[0].sparsify.ops) == 1
    assert torch.allclose(op.forward(x), x.reshape(3, 2))
    assert torch.allclose(op.backward(grad), grad.reshape(2, 3))
    assert torch.allclose(op.jacobian(), expected_jacobian)


def test_permuteop_constructs_sparse_matrix():
    op = PermuteOp(torch.Size([2, 3, 4]), (2, 0, 1))
    x = torch.arange(24.0).reshape(2, 3, 4)
    grad = torch.arange(24.0).reshape(4, 2, 3) / 3.0
    expected_jacobian = _expected_jacobian(op.input_shape, op.output_shape, lambda t: t.permute(2, 0, 1))

    assert isinstance(op.tensor, MultiCOOTensorSum)
    assert len(op.tensor.terms) == 1
    assert all(sparsify.is_md_eye() for sparsify in op.tensor.terms[0].sparsify.ops)
    assert torch.allclose(op.forward(x), x.permute(2, 0, 1))
    assert torch.allclose(op.backward(grad), grad.permute(1, 2, 0))
    assert torch.allclose(op.jacobian(), expected_jacobian)


def test_expandop_constructs_sparse_matrix():
    op = ExpandOp(torch.Size([1, 3, 1]), torch.Size([4, 3, 5]))
    x = torch.arange(3.0).reshape(1, 3, 1)
    grad = torch.arange(60.0).reshape(4, 3, 5) / 7.0
    expected_jacobian = _expected_jacobian(op.input_shape, op.output_shape, lambda t: t.expand(4, 3, 5))

    assert isinstance(op.tensor, MultiCOOTensorSum)
    assert len(op.tensor.terms) == 1
    assert len(op.tensor.terms[0].sparsify.ops) == 3
    assert torch.allclose(op.forward(x), x.expand(4, 3, 5))
    assert torch.allclose(op.backward(grad), grad.sum((0, 2), keepdim=True))
    assert torch.allclose(op.jacobian(), expected_jacobian)


def test_getsliceop_constructs_sparse_matrix():
    op = GetSliceOp(torch.Size([4, 5]), [[slice(1, 4)], [slice(0, 5)]])
    x = torch.arange(20.0).reshape(4, 5)
    grad = torch.arange(15.0).reshape(3, 5) - 2.0
    expected_jacobian = _expected_jacobian(op.input_shape, op.output_shape, lambda t: t[1:4, :])

    assert isinstance(op.tensor, MultiCOOTensorSum)
    assert len(op.tensor.terms) == 1
    assert len(op.tensor.terms[0].sparsify.ops) == 2
    assert torch.allclose(op.forward(x), x[1:4, :])
    assert torch.allclose(op.backward(grad), torch.cat([torch.zeros(1, 5), grad], dim=0))
    assert torch.allclose(op.jacobian(), expected_jacobian)


def test_getindicesop_constructs_sparse_matrix():
    indices = torch.tensor([3, 1])
    op = GetIndicesOp(torch.Size([4, 3]), dim=0, indices=indices, added_shape=torch.Size([2]))
    x = torch.arange(12.0).reshape(4, 3)
    grad = torch.arange(6.0).reshape(2, 3) - 1.0
    expected_jacobian = _expected_jacobian(op.input_shape, op.output_shape, lambda t: t.index_select(0, indices))

    expected_backward = torch.zeros(4, 3)
    expected_backward.index_add_(0, indices, grad)
    assert isinstance(op.tensor, MultiCOOTensorSum)
    assert len(op.tensor.terms) == 1
    assert len(op.tensor.terms[0].sparsify.ops) == 2
    assert torch.allclose(op.forward(x), x.index_select(0, indices))
    assert torch.allclose(op.backward(grad), expected_backward)
    assert torch.allclose(op.jacobian(), expected_jacobian)


def _assert_sparse_linearop(op, x, grad, forward_fn, backward_expected):
    expected_jacobian = _expected_jacobian(op.input_shape, op.output_shape, forward_fn)

    assert isinstance(op.tensor, MultiCOOTensorSum)
    assert len(op.tensor.terms) >= 1
    assert torch.allclose(op.forward(x), forward_fn(x))
    assert torch.allclose(op.backward(grad), backward_expected)
    assert torch.allclose(op.jacobian(), expected_jacobian)


def test_reshape_family_sparse_matrices():
    flatten = FlattenOp(torch.Size([2, 3, 4]), 1, 2)
    _assert_sparse_linearop(
        flatten,
        torch.arange(24.0).reshape(2, 3, 4),
        torch.arange(24.0).reshape(2, 12),
        lambda t: t.flatten(1, 2),
        torch.arange(24.0).reshape(2, 3, 4),
    )

    unflatten = UnflattenOp(torch.Size([2, 12]), 1, (3, 4))
    _assert_sparse_linearop(
        unflatten,
        torch.arange(24.0).reshape(2, 12),
        torch.arange(24.0).reshape(2, 3, 4),
        lambda t: t.unflatten(1, (3, 4)),
        torch.arange(24.0).reshape(2, 12),
    )

    squeeze = SqueezeOp(torch.Size([2, 1, 4]), 1)
    _assert_sparse_linearop(
        squeeze,
        torch.arange(8.0).reshape(2, 1, 4),
        torch.arange(8.0).reshape(2, 4),
        lambda t: t.squeeze(1),
        torch.arange(8.0).reshape(2, 1, 4),
    )

    unsqueeze = UnsqueezeOp(torch.Size([2, 4]), 1)
    _assert_sparse_linearop(
        unsqueeze,
        torch.arange(8.0).reshape(2, 4),
        torch.arange(8.0).reshape(2, 1, 4),
        lambda t: t.unsqueeze(1),
        torch.arange(8.0).reshape(2, 4),
    )


def test_transpose_and_misc_shape_sparse_matrices():
    transpose = TransposeOp(torch.Size([2, 3, 4]), 0, 2)
    grad = torch.arange(24.0).reshape(4, 3, 2)
    _assert_sparse_linearop(
        transpose,
        torch.arange(24.0).reshape(2, 3, 4),
        grad,
        lambda t: t.transpose(0, 2),
        grad.transpose(0, 2),
    )

    repeat = RepeatOp(torch.Size([2, 3]), (2, 3))
    grad = torch.arange(36.0).reshape(4, 9)
    _assert_sparse_linearop(
        repeat,
        torch.arange(6.0).reshape(2, 3),
        grad,
        lambda t: t.repeat(2, 3),
        grad.reshape(2, 2, 3, 3).sum((0, 2)),
    )

    tile = TileOp(torch.Size([2, 3]), (2, 3))
    _assert_sparse_linearop(
        tile,
        torch.arange(6.0).reshape(2, 3),
        grad,
        lambda t: torch.tile(t, (2, 3)),
        grad.reshape(2, 2, 3, 3).sum((0, 2)),
    )

    flip = FlipOp(torch.Size([2, 3]), (1,))
    grad = torch.arange(6.0).reshape(2, 3)
    _assert_sparse_linearop(
        flip,
        torch.arange(6.0).reshape(2, 3),
        grad,
        lambda t: t.flip((1,)),
        grad.flip((1,)),
    )

    roll = RollOp(torch.Size([2, 3]), 1, 1)
    _assert_sparse_linearop(
        roll,
        torch.arange(6.0).reshape(2, 3),
        grad,
        lambda t: t.roll(1, 1),
        grad.roll(-1, 1),
    )


def test_diagop_sparse_matrices():
    diag_create = DiagOp(torch.Size([3]))
    grad = torch.arange(9.0).reshape(3, 3)
    _assert_sparse_linearop(
        diag_create,
        torch.arange(3.0),
        grad,
        lambda t: t.diag(),
        grad.diag(),
    )

    diag_extract = DiagOp(torch.Size([3, 3]))
    grad = torch.arange(3.0)
    _assert_sparse_linearop(
        diag_extract,
        torch.arange(9.0).reshape(3, 3),
        grad,
        lambda t: t.diag(),
        torch.diag(grad),
    )


def test_setsliceop_constructs_sparse_matrix():
    op = SetSliceOp(torch.Size([4, 5]), [[slice(1, 4)], [slice(0, 5)]])
    x = torch.arange(15.0).reshape(3, 5)
    grad = torch.arange(20.0).reshape(4, 5)

    def forward_fn(t):
        result = torch.zeros(4, 5)
        result[1:4, :] = t
        return result

    _assert_sparse_linearop(op, x, grad, forward_fn, grad[1:4, :])


def test_gather_scatter_construct_sparse_matrices():
    index = torch.tensor([[0, 2], [1, 0]])
    gather = GatherOp(torch.Size([3, 2]), 0, index)
    grad = torch.arange(4.0).reshape(2, 2)
    expected_backward = torch.zeros(3, 2)
    expected_backward.scatter_add_(0, index, grad)
    _assert_sparse_linearop(
        gather,
        torch.arange(6.0).reshape(3, 2),
        grad,
        lambda t: torch.gather(t, 0, index),
        expected_backward,
    )

    scatter = ScatterOp(torch.Size([2, 2]), 0, index, torch.Size([3, 2]))
    x = torch.arange(4.0).reshape(2, 2)
    grad = torch.arange(6.0).reshape(3, 2)

    def scatter_forward(t):
        result = torch.zeros(3, 2)
        result.scatter_add_(0, index, t)
        return result

    _assert_sparse_linearop(scatter, x, grad, scatter_forward, torch.gather(grad, 0, index))


def test_setindicesop_constructs_sparse_matrix():
    indices = torch.tensor([[2, 0], [1, 2]])
    op = SetIndicesOp(torch.Size([3, 4]), 0, indices, torch.Size([2, 2]))
    x = torch.arange(16.0).reshape(2, 2, 4)
    grad = torch.arange(12.0).reshape(3, 4)

    def forward_fn(t):
        result = torch.zeros(3, 4)
        result.index_add_(0, indices.reshape(-1), t.flatten(0, 1))
        return result

    expected_backward = grad.index_select(0, indices.reshape(-1)).unflatten(0, (2, 2))
    _assert_sparse_linearop(op, x, grad, forward_fn, expected_backward)
