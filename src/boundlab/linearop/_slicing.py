"""Slice-based indexing LinearOp implementations."""

import torch

from boundlab.linearop._base import DEBUG_LINEAR_OP, LinearOp, LinearOpFlags
from boundlab.linearop._debug import jacobian_from_function
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


def _is_full(dim_slices: list[slice], dim_size: int) -> bool:
    return len(dim_slices) == 1 and dim_slices[0].start == 0 and dim_slices[0].stop == dim_size


def _slice_tensor(input_shape: torch.Size, output_shape: torch.Size, all_slices: list[list[slice]], set_mode: bool):
    input_dims = make_input_dims(input_shape)
    output_dims = make_output_dims(output_shape)
    ops = []
    for axis, slices in enumerate(all_slices):
        in_dim = input_dims[axis]
        out_dim = output_dims[axis]
        if not set_mode:
            if _is_full(slices, input_shape[axis]):
                assert input_shape[axis] == output_shape[axis], f"Full slice on axis {axis} requires input and output sizes to match, got {input_shape[axis]} and {output_shape[axis]}"
                inner = Dim(int(input_shape[axis]), 500.0 + axis, f"k{axis}")
                ops.append(COOSparsify.md_eye(inner, [out_dim, in_dim]))
                continue

            edge = Dim(output_shape[axis], 500.0 + axis, f"k{axis}")
            input_pos = torch.cat([torch.arange(s.start, s.stop, dtype=torch.long) for s in slices], dim=0)
            assert input_pos.shape[0] == out_dim.length, f"Number of output indices must match output size on axis {axis}, got {input_pos.numel()} and {output_shape[axis]}"
            length = out_dim.length
            output_pos = None
        else:
            if _is_full(slices, output_shape[axis]):
                assert input_shape[axis] == output_shape[axis], f"Full slice on axis {axis} requires input and output sizes to match, got {input_shape[axis]} and {output_shape[axis]}"
                inner = Dim(int(input_shape[axis]), 500.0 + axis, f"k{axis}")
                ops.append(COOSparsify.md_eye(inner, [in_dim, out_dim]))
                continue

            edge = Dim(input_shape[axis], 500.0 + axis, f"k{axis}")
            output_pos = torch.cat([torch.arange(s.start, s.stop, dtype=torch.long) for s in slices], dim=0)
            assert output_pos.shape[0] == in_dim.length, f"Number of input indices must match input size on axis {axis}, got {output_pos.numel()} and {input_shape[axis]}"
            length = in_dim.length
            input_pos = None
        ops.append(
            COOSparsify(
                edge,
                TorchTable(
                    columns=[out_dim, in_dim],
                    data=[output_pos, input_pos],
                    length=length,
                    is_unique=True,
                    is_sorted=True,
                ),
            )
        )
    tensor = MultiCOOTensor(TN(factors=[]), MultiCOOSparsify(ops))
    return MultiCOOTensorSum([tensor]), input_dims, output_dims


def _get_slice_debug(x: torch.Tensor, indices: list[torch.Tensor]) -> torch.Tensor:
    for axis, idx in enumerate(indices):
        x = x.index_select(axis, idx.to(x.device))
    return x


def _set_slice_debug(x: torch.Tensor, output_shape: torch.Size, indices: list[torch.Tensor]) -> torch.Tensor:
    result = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
    mesh = torch.meshgrid(*(idx.to(x.device) for idx in indices), indexing="ij")
    result[mesh] = x
    return result


class GetSliceOp(LinearOp):
    def __init__(self, input_shape: torch.Size, slices: list[list[slice]]):
        input_shape = torch.Size(input_shape)
        assert len(input_shape) == len(slices)
        self.slices = _normalize_slices(slices, input_shape)
        output_shape = torch.Size(_output_size(s) for s in self.slices)

        tensor, input_dims, output_dims = _slice_tensor(input_shape, output_shape, self.slices, set_mode=False)
        debug_jacobian = jacobian_from_function(input_shape, output_shape, lambda x: _get_slice_debug(x, self._indices)) if DEBUG_LINEAR_OP else None
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
        input_shape = torch.Size(_output_size(s) for s in self.slices)
        tensor, input_dims, output_dims = _slice_tensor(input_shape, output_shape, self.slices, set_mode=True)
        debug_jacobian = jacobian_from_function(input_shape, output_shape, lambda x: _set_slice_debug(x, output_shape, self._indices)) if DEBUG_LINEAR_OP else None
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
