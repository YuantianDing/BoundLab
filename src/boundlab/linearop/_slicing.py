"""Slice-based indexing LinearOp implementations."""

import torch

from boundlab.linearop._base import DEBUG_LinearOp, LinearOp, LinearOpFlags
from boundlab.linearop._sparse import make_input_dims, make_output_dims
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import TN


def _normalize_slices(slices: list[list[slice]], shape: torch.Size) -> list[list[slice]]:
    result = []
    for d, dim_slices in enumerate(slices):
        normalized = []
        for s in dim_slices:
            start, stop, step = s.indices(shape[d])
            assert step == 1, f"slice steps other than 1 are not supported here, got {step} in dim {d}"
            normalized.append(slice(start, stop))
        result.append(normalized)
    return result


def _output_size(dim_slices: list[slice]) -> int:
    return sum(s.stop - s.start for s in dim_slices)


def _slice_indices(dim_slices: list[slice]) -> torch.Tensor:
    if not dim_slices:
        return torch.empty(0, dtype=torch.long)
    return torch.cat([torch.arange(s.start, s.stop, dtype=torch.long) for s in dim_slices])


def _is_full(dim_slices: list[slice], dim_size: int) -> bool:
    return len(dim_slices) == 1 and dim_slices[0].start == 0 and dim_slices[0].stop == dim_size


def _slice_tensor(input_shape: torch.Size, output_shape: torch.Size, indices: list[torch.Tensor], set_mode: bool):
    input_dims = make_input_dims(input_shape)
    output_dims = make_output_dims(output_shape)
    ops = []
    for axis, idx in enumerate(indices):
        in_dim = input_dims[axis]
        out_dim = output_dims[axis]
        if idx.numel() == input_shape[axis] == output_shape[axis] and torch.equal(idx, torch.arange(input_shape[axis])):
            inner = Dim(int(input_shape[axis]), 500.0 + axis, f"k{axis}")
            ops.append(COOSparsify.md_eye(inner, [out_dim, in_dim]))
            continue

        edge = Dim(int(idx.numel()), 500.0 + axis, f"k{axis}")
        input_pos = torch.arange(idx.numel(), dtype=torch.long)
        output_pos = idx.to(torch.long).contiguous()
        if not set_mode:
            input_pos, output_pos = output_pos, input_pos
        ops.append(
            COOSparsify(
                edge,
                TorchTable(
                    columns=[out_dim, in_dim],
                    data=[output_pos, input_pos],
                    length=int(idx.numel()),
                ),
            )
        )
    tensor = MultiCOOTensor(TN(factors=[]), MultiCOOSparsify(ops))
    return MultiCOOTensorSum([tensor]), input_dims, output_dims


class GetSliceOp(LinearOp):
    def __init__(self, input_shape: torch.Size, slices: list[list[slice]]):
        input_shape = torch.Size(input_shape)
        assert len(input_shape) == len(slices)
        self.slices = _normalize_slices(slices, input_shape)
        self._indices = [_slice_indices(s) for s in self.slices]
        output_shape = torch.Size(_output_size(s) for s in self.slices)

        tensor, input_dims, output_dims = _slice_tensor(input_shape, output_shape, self._indices, set_mode=False)
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

    def __matmul__(self, other):
        if isinstance(other, GetSliceOp):
            assert self.input_shape == other.output_shape
            return GetSliceOp(other.input_shape, _compose_get_slices(self.slices, other.slices))
        return super().__matmul__(other)

    def __str__(self):
        parts = []
        for dim_slices in self.slices:
            if len(dim_slices) == 1:
                s = dim_slices[0]
                parts.append(f"{s.start}:{s.stop}")
            else:
                parts.append("[" + ",".join(f"{s.start}:{s.stop}" for s in dim_slices) + "]")
        return f"<getslice {','.join(parts)}>"


class SetSliceOp(LinearOp):
    def __init__(self, output_shape: torch.Size, slices: list[list[slice]]):
        output_shape = torch.Size(output_shape)
        assert len(output_shape) == len(slices)
        self.slices = _normalize_slices(slices, output_shape)
        self._indices = [_slice_indices(s) for s in self.slices]
        input_shape = torch.Size(_output_size(s) for s in self.slices)
        tensor, input_dims, output_dims = _slice_tensor(input_shape, output_shape, self._indices, set_mode=True)
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

    def __matmul__(self, other):
        if isinstance(other, SetSliceOp):
            assert self.input_shape == other.output_shape
            return SetSliceOp(self.output_shape, _compose_set_slices(self.slices, other.slices))
        return super().__matmul__(other)

    def __str__(self):
        parts = []
        for dim_slices in self.slices:
            if len(dim_slices) == 1:
                s = dim_slices[0]
                parts.append(f"{s.start}:{s.stop}")
            else:
                parts.append("[" + ",".join(f"{s.start}:{s.stop}" for s in dim_slices) + "]")
        return f"<setslice {','.join(parts)}>"


def _compose_get_slices(outer_slices, inner_slices):
    return [_compose_dim_slices(outer, inner) for outer, inner in zip(outer_slices, inner_slices)]


def _compose_set_slices(outer_slices, inner_slices):
    return [_compose_dim_slices(inner, outer) for outer, inner in zip(outer_slices, inner_slices)]


def _compose_dim_slices(outer: list[slice], inner: list[slice]) -> list[slice]:
    result = []
    for o_slice in outer:
        pos = 0
        for i_slice in inner:
            length = i_slice.stop - i_slice.start
            inter_start = max(o_slice.start, pos)
            inter_stop = min(o_slice.stop, pos + length)
            if inter_start < inter_stop:
                result.append(slice(i_slice.start + inter_start - pos, i_slice.start + inter_stop - pos))
            pos += length
    return _merge_adjacent_slices(result)


def _merge_adjacent_slices(slices: list[slice]) -> list[slice]:
    if not slices:
        return [slice(0, 0)]
    result = [slices[0]]
    for s in slices[1:]:
        if s.start <= result[-1].stop:
            result[-1] = slice(result[-1].start, max(result[-1].stop, s.stop))
        else:
            result.append(s)
    return result
