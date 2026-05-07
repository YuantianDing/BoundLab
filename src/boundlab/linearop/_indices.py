"""Indexing LinearOp implementations for bound propagation."""

import torch

from boundlab.linearop._base import DEBUG_LinearOp, LinearOp, LinearOpFlags
from boundlab.linearop._reshape import SqueezeOp
from boundlab.linearop._sparse import all_coords, tensor_from_edges, tensor_from_output_map
from boundlab.linearop._slicing import GetSliceOp, SetSliceOp
from boundlab.linearop._indexing import GetIndicesOp, SetIndicesOp


class GatherOp(LinearOp):
    def __init__(self, input_shape: torch.Size, dim: int, index: torch.Tensor):
        input_shape = torch.Size(input_shape)
        self.dim = dim
        self.index = index.to(torch.long)
        output_shape = torch.Size(index.shape)

        def input_from_output(out):
            inp = out.clone()
            inp[:, dim] = self.index.reshape(-1)[_ravel_like(out, output_shape)]
            return inp

        tensor, input_dims, output_dims = tensor_from_output_map(input_shape, output_shape, input_from_output)
        debug_jacobian = tensor.to_dense().expand(output_dims + input_dims) if DEBUG_LinearOp else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LinearOp:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __str__(self):
        return f"<gather dim={self.dim} index.shape={list(self.index.shape)}>"


class ScatterOp(LinearOp):
    def __init__(self, input_shape: torch.Size, dim: int, index: torch.Tensor, output_shape: torch.Size):
        input_shape = torch.Size(input_shape)
        output_shape = torch.Size(output_shape)
        self.dim = dim
        self.index = index.to(torch.long)
        assert self.index.shape == input_shape
        input_coords = all_coords(input_shape)
        output_coords = input_coords.clone()
        output_coords[:, dim] = self.index.reshape(-1)[_ravel_like(input_coords, input_shape)]
        tensor, input_dims, output_dims = tensor_from_edges(input_shape, output_shape, input_coords, output_coords)
        debug_jacobian = tensor.to_dense().expand(output_dims + input_dims) if DEBUG_LinearOp else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LinearOp:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __str__(self):
        return f"<scatter dim={self.dim} index.shape={list(self.index.shape)}>"


def _ravel_like(coords: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
    for axis, size in enumerate(shape):
        result = result * int(size) + coords[:, axis]
    return result


def narrow_indices(ndim: int, dim: int, start: int, length: int) -> tuple:
    indices = [slice(None)] * ndim
    indices[dim] = slice(start, start + length)
    return tuple(indices)


def select_indices(ndim: int, dim: int, index: int) -> tuple:
    indices = [slice(None)] * ndim
    indices[dim] = index
    return tuple(indices)


def pad_indices(input_shape: torch.Size, pad_spec: list[int]) -> tuple:
    ndim = len(input_shape)
    indices = []
    for d in range(ndim):
        d_rev = ndim - 1 - d
        if 2 * d_rev + 1 < len(pad_spec):
            before = pad_spec[2 * d_rev]
            indices.append(slice(before, before + input_shape[d]))
        else:
            indices.append(slice(None))
    return tuple(indices)


def pad_output_shape(input_shape: torch.Size, pad_spec: list[int]) -> torch.Size:
    output = list(input_shape)
    ndim = len(input_shape)
    for d in range(ndim):
        d_rev = ndim - 1 - d
        if 2 * d_rev + 1 < len(pad_spec):
            output[d] += pad_spec[2 * d_rev] + pad_spec[2 * d_rev + 1]
    return torch.Size(output)


def make_get_slices(input_shape: torch.Size, indices) -> list[list[slice]]:
    if not isinstance(indices, tuple):
        indices = (indices,)
    ndim = len(input_shape)
    normalized = []
    idx_pos = 0
    saw_ellipsis = False
    while len(normalized) < ndim:
        if idx_pos >= len(indices):
            normalized.append([slice(0, input_shape[len(normalized)])])
            continue
        idx = indices[idx_pos]
        if idx is Ellipsis:
            if saw_ellipsis:
                raise ValueError("Only one Ellipsis allowed")
            saw_ellipsis = True
            remaining = ndim - (len(indices) - 1 - idx_pos) - len(normalized)
            for _ in range(remaining):
                normalized.append([slice(0, input_shape[len(normalized)])])
            idx_pos += 1
            continue
        dim = len(normalized)
        if isinstance(idx, int):
            if idx < 0:
                idx += input_shape[dim]
            normalized.append([slice(idx, idx + 1)])
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(input_shape[dim])
            if step == 1:
                normalized.append([slice(start, stop)])
            else:
                positions = list(range(start, stop, step))
                normalized.append([slice(p, p + 1) for p in positions])
        else:
            raise ValueError(f"Unsupported index type: {type(idx)}")
        idx_pos += 1
    return normalized


def make_set_slices(output_shape: torch.Size, indices) -> list[list[slice]]:
    return make_get_slices(output_shape, indices)


def get_int_dims(indices) -> list[int]:
    if not isinstance(indices, tuple):
        indices = (indices,)
    result = []
    pos = 0
    for idx in indices:
        if idx is Ellipsis:
            continue
        if isinstance(idx, int):
            result.append(pos)
        pos += 1
    return result


def NarrowOp(input_shape: torch.Size, dim: int, start: int, length: int):
    return GetSliceOp(input_shape, make_get_slices(input_shape, narrow_indices(len(input_shape), dim, start, length)))


def SelectOp(input_shape: torch.Size, dim: int, index: int):
    result = GetSliceOp(input_shape, make_get_slices(input_shape, select_indices(len(input_shape), dim, index)))
    return SqueezeOp(result.output_shape, dim) @ result


def GetItemOp(input_shape: torch.Size, indices):
    result = GetSliceOp(input_shape, make_get_slices(input_shape, indices))
    for dim in sorted(get_int_dims(indices), reverse=True):
        result = SqueezeOp(result.output_shape, dim) @ result
    return result


def PadOp(input_shape: torch.Size, pad_spec: list[int]):
    output_shape = pad_output_shape(input_shape, pad_spec)
    return SetSliceOp(output_shape, make_set_slices(output_shape, pad_indices(input_shape, pad_spec)))
