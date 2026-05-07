"""Index-tensor-based LinearOp implementations."""

import torch

from boundlab.linearop._base import DEBUG_LINEAR_OP, LinearOp, LinearOpFlags
from boundlab.linearop._debug import jacobian_from_function
from boundlab.linearop._sparse import all_coords, make_input_dims, make_output_dims
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import TN


def _indices_tensor(input_shape: torch.Size, output_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size, set_mode: bool):
    input_dims = make_input_dims(input_shape)
    output_dims = make_output_dims(output_shape)
    n_added = len(added_shape)
    ops = []

    for axis in range(dim):
        inner = Dim(int(input_shape[axis]), 500.0 + axis, f"k{axis}")
        ops.append(COOSparsify.md_eye(inner, [output_dims[axis], input_dims[axis]]))

    for axis in range(dim + 1, len(output_shape if set_mode else input_shape)):
        input_axis = axis + n_added - 1 if set_mode else axis
        output_axis = axis if set_mode else axis + n_added - 1
        inner = Dim(int(input_shape[input_axis]), 500.0 + output_axis, f"k{output_axis}")
        ops.append(COOSparsify.md_eye(inner, [output_dims[output_axis], input_dims[input_axis]]))

    edge = Dim(int(indices.numel()), 500.0 + dim, f"k{dim}")
    coords = all_coords(added_shape)
    if set_mode:
        columns = [output_dims[dim]] + [input_dims[dim + axis] for axis in range(n_added)]
        data = [indices.reshape(-1).to(torch.long).contiguous()] + [
            coords[:, axis].contiguous() for axis in range(n_added)
        ]
    else:
        columns = [output_dims[dim + axis] for axis in range(n_added)] + [input_dims[dim]]
        data = [coords[:, axis].contiguous() for axis in range(n_added)] + [
            indices.reshape(-1).to(torch.long).contiguous()
        ]
    ops.append(COOSparsify(edge, TorchTable(columns=columns, data=data, length=int(indices.numel()))))

    tensor = MultiCOOTensor(TN(factors=[]), MultiCOOSparsify(ops))
    return MultiCOOTensorSum([tensor]), input_dims, output_dims


def _get_indices_debug(x: torch.Tensor, dim: int, indices: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
    return x.index_select(dim, indices.reshape(-1).to(x.device)).reshape(output_shape)


def _set_indices_debug(x: torch.Tensor, output_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size) -> torch.Tensor:
    input_coords = all_coords(torch.Size(x.shape)).to(x.device)
    output_coords = input_coords.clone()
    flat_indices = indices.reshape(-1).to(x.device)
    added_coords = all_coords(added_shape).to(x.device)
    added_ravel = torch.zeros(added_coords.shape[0], dtype=torch.long, device=x.device)
    for axis, size in enumerate(added_shape):
        added_ravel = added_ravel * int(size) + added_coords[:, axis]
    input_added_coords = input_coords[:, dim:dim + len(added_shape)]
    input_ravel = torch.zeros(input_coords.shape[0], dtype=torch.long, device=x.device)
    for axis, size in enumerate(added_shape):
        input_ravel = input_ravel * int(size) + input_added_coords[:, axis]
    output_coords[:, dim] = flat_indices[input_ravel]
    if len(added_shape) > 1:
        output_coords = torch.cat([output_coords[:, :dim + 1], output_coords[:, dim + len(added_shape):]], dim=1)
    result = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
    result.index_put_(tuple(output_coords[:, axis] for axis in range(output_coords.shape[1])), x.reshape(-1), accumulate=True)
    return result


class GetIndicesOp(LinearOp):
    def __init__(self, input_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size):
        input_shape = torch.Size(input_shape)
        self.dim = dim
        self.indices = indices.to(torch.long)
        self.added_shape = torch.Size(added_shape)
        assert self.indices.shape == self.added_shape
        output_shape = torch.Size(list(input_shape[:dim]) + list(self.added_shape) + list(input_shape[dim + 1:]))
        n_added = len(self.added_shape)
        tensor, input_dims, output_dims = _indices_tensor(input_shape, output_shape, dim, self.indices, self.added_shape, set_mode=False)
        debug_jacobian = jacobian_from_function(input_shape, output_shape, lambda x: _get_indices_debug(x, dim, self.indices, output_shape)) if DEBUG_LINEAR_OP else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __matmul__(self, other):
        if isinstance(other, GetIndicesOp) and self.dim == other.dim:
            new_indices = other.indices.reshape(-1)[self.indices.reshape(-1)].reshape(self.added_shape)
            return GetIndicesOp(other.input_shape, self.dim, new_indices, self.added_shape)
        return super().__matmul__(other)

    def __str__(self):
        return f"<getindices dim={self.dim} added={list(self.added_shape)}>"


class SetIndicesOp(LinearOp):
    def __init__(self, output_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size):
        output_shape = torch.Size(output_shape)
        self.dim = dim
        self.indices = indices.to(torch.long)
        self.added_shape = torch.Size(added_shape)
        assert self.indices.shape == self.added_shape
        input_shape = torch.Size(list(output_shape[:dim]) + list(self.added_shape) + list(output_shape[dim + 1:]))
        tensor, input_dims, output_dims = _indices_tensor(input_shape, output_shape, dim, self.indices, self.added_shape, set_mode=True)
        debug_jacobian = jacobian_from_function(input_shape, output_shape, lambda x: _set_indices_debug(x, output_shape, dim, self.indices, self.added_shape)) if DEBUG_LINEAR_OP else None
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
        return f"<setindices dim={self.dim} added={list(self.added_shape)}>"
