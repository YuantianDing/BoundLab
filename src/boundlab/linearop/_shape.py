"""Miscellaneous shape LinearOp implementations."""

import torch

from boundlab.linearop._base import DEBUG_LINEAR_OP, LinearOp, LinearOpFlags
from boundlab.linearop._debug import jacobian_from_function
from boundlab.linearop._sparse import make_input_dims, make_output_dims
from boundlab.linearop._reshape import ReshapeOp, FlattenOp, UnflattenOp, SqueezeOp, UnsqueezeOp, _meta_output_shape
from boundlab.linearop._permute import PermuteOp, TransposeOp
from boundlab.linearop._expand import ExpandOp
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import TN


def _axis_op(input_dim, output_dim, mapping: torch.Tensor, ordering: float, name: str):
    mapping = mapping.to(torch.long).contiguous()
    if input_dim is not None and mapping.numel() == input_dim.length == output_dim.length and torch.equal(mapping, torch.arange(output_dim.length)):
        return COOSparsify.md_eye(Dim(output_dim.length, ordering, name), [output_dim, input_dim])

    edge = Dim(int(mapping.numel()), ordering, name)
    columns = [output_dim]
    data = [torch.arange(mapping.numel(), dtype=torch.long)]
    if input_dim is not None:
        columns.append(input_dim)
        data.append(mapping)
    return COOSparsify(edge, TorchTable(columns=columns, data=data, length=int(mapping.numel())))


def _tensor_from_ops(input_dims, output_dims, ops):
    tensor = MultiCOOTensor(TN(factors=[]), MultiCOOSparsify(ops))
    return MultiCOOTensorSum([tensor]), input_dims, output_dims


class RepeatOp(LinearOp):
    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        input_shape = torch.Size(input_shape)
        self.sizes = tuple(sizes)
        self._n_pad = len(self.sizes) - len(input_shape)
        assert self._n_pad >= 0
        self._padded_input_shape = torch.Size([1] * self._n_pad + list(input_shape))
        output_shape = torch.Size(s * r for s, r in zip(self._padded_input_shape, self.sizes))

        input_dims = make_input_dims(input_shape)
        output_dims = make_output_dims(output_shape)
        ops = []
        for axis, output_dim in enumerate(output_dims):
            if axis < self._n_pad:
                mapping = torch.arange(output_shape[axis], dtype=torch.long)
                ops.append(_axis_op(None, output_dim, mapping, 500.0 + axis, f"k{axis}"))
                continue
            input_axis = axis - self._n_pad
            mapping = torch.arange(output_shape[axis], dtype=torch.long) % input_shape[input_axis]
            ops.append(_axis_op(input_dims[input_axis], output_dim, mapping, 500.0 + axis, f"k{axis}"))
        tensor, input_dims, output_dims = _tensor_from_ops(input_dims, output_dims, ops)
        debug_jacobian = jacobian_from_function(input_shape, output_shape, lambda x: x.repeat(self.sizes)) if DEBUG_LINEAR_OP else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __str__(self):
        return f"<repeat {list(self.sizes)}>"


class TileOp(RepeatOp):
    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        self.tile_sizes = tuple(sizes)
        n_pad = len(input_shape) - len(sizes)
        if n_pad > 0:
            sizes = (1,) * n_pad + tuple(sizes)
        super().__init__(input_shape, sizes)

    def __str__(self):
        return f"<tile {list(self.tile_sizes)}>"


class FlipOp(LinearOp):
    def __init__(self, input_shape: torch.Size, dims):
        input_shape = torch.Size(input_shape)
        self.dims = tuple(dims) if not isinstance(dims, int) else (dims,)

        input_dims = make_input_dims(input_shape)
        output_dims = make_output_dims(input_shape)
        flip_dims = {d if d >= 0 else len(input_shape) + d for d in self.dims}
        ops = []
        for axis, (input_dim, output_dim) in enumerate(zip(input_dims, output_dims)):
            mapping = torch.arange(input_shape[axis], dtype=torch.long)
            if axis in flip_dims:
                mapping = input_shape[axis] - 1 - mapping
            ops.append(_axis_op(input_dim, output_dim, mapping, 500.0 + axis, f"k{axis}"))
        tensor, input_dims, output_dims = _tensor_from_ops(input_dims, output_dims, ops)
        debug_jacobian = jacobian_from_function(input_shape, input_shape, lambda x: torch.flip(x, tuple(flip_dims))) if DEBUG_LINEAR_OP else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __str__(self):
        return f"<flip {self.dims}>"


class RollOp(LinearOp):
    def __init__(self, input_shape: torch.Size, shifts, dims):
        input_shape = torch.Size(input_shape)
        self.shifts = shifts
        self.dims = dims
        dims_tuple = (dims,) if isinstance(dims, int) else tuple(dims)
        shifts_tuple = (shifts,) if isinstance(shifts, int) else tuple(shifts)
        self._inv_shifts = -shifts if isinstance(shifts, int) else [-s for s in shifts]

        shift_by_dim = {
            dim if dim >= 0 else len(input_shape) + dim: shift
            for shift, dim in zip(shifts_tuple, dims_tuple)
        }
        input_dims = make_input_dims(input_shape)
        output_dims = make_output_dims(input_shape)
        ops = []
        for axis, (input_dim, output_dim) in enumerate(zip(input_dims, output_dims)):
            mapping = torch.arange(input_shape[axis], dtype=torch.long)
            if axis in shift_by_dim:
                mapping = (mapping - shift_by_dim[axis]) % input_shape[axis]
            ops.append(_axis_op(input_dim, output_dim, mapping, 500.0 + axis, f"k{axis}"))
        tensor, input_dims, output_dims = _tensor_from_ops(input_dims, output_dims, ops)
        debug_jacobian = jacobian_from_function(input_shape, input_shape, lambda x: torch.roll(x, shifts_tuple, dims_tuple)) if DEBUG_LINEAR_OP else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __str__(self):
        return f"<roll {self.shifts} {self.dims}>"


class DiagOp(LinearOp):
    def __init__(self, input_shape: torch.Size, diagonal: int = 0):
        input_shape = torch.Size(input_shape)
        self.diagonal = diagonal
        self._input_ndim = len(input_shape)
        output_shape = _meta_output_shape(lambda x: x.diag(diagonal), input_shape)
        if len(input_shape) == 1:
            n = input_shape[0]
            k = torch.arange(n, dtype=torch.long)
            input_coords = k[:, None]
            output_coords = torch.empty((n, 2), dtype=torch.long)
            if diagonal >= 0:
                output_coords[:, 0] = k
                output_coords[:, 1] = k + diagonal
            else:
                output_coords[:, 0] = k - diagonal
                output_coords[:, 1] = k
        else:
            n = output_shape[0]
            k = torch.arange(n, dtype=torch.long)
            output_coords = k[:, None]
            input_coords = torch.empty((n, 2), dtype=torch.long)
            if diagonal >= 0:
                input_coords[:, 0] = k
                input_coords[:, 1] = k + diagonal
            else:
                input_coords[:, 0] = k - diagonal
                input_coords[:, 1] = k
        input_dims = make_input_dims(input_shape)
        output_dims = make_output_dims(output_shape)
        if diagonal == 0 and all(dim.length == int(input_coords.shape[0]) for dim in input_dims + output_dims):
            op = COOSparsify.md_eye(Dim(int(input_coords.shape[0]), 500.0, "k_diag"), output_dims + input_dims)
        else:
            edge = Dim(int(input_coords.shape[0]), 500.0, "k_diag")
            data = [
                *(output_coords[:, axis].contiguous() for axis in range(len(output_dims))),
                *(input_coords[:, axis].contiguous() for axis in range(len(input_dims))),
            ]
            op = COOSparsify(edge, TorchTable(columns=output_dims + input_dims, data=data, length=int(input_coords.shape[0])))
        tensor, input_dims, output_dims = _tensor_from_ops(input_dims, output_dims, [op])
        debug_jacobian = jacobian_from_function(input_shape, output_shape, lambda x: x.diag(diagonal)) if DEBUG_LINEAR_OP else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __str__(self):
        return f"<diag {self.diagonal}>"
