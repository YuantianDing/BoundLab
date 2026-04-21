"""Slice-based indexing LinearOp implementations.

``GetSliceOp`` and ``SetSliceOp`` use a structured ``list[list[slice]]``
format where each dimension has a list of non-overlapping slices.
``len(input_shape) == len(slices)`` is enforced.
"""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags, ComposedOp
from boundlab.utils import merge_name


def _normalize_slices(slices: list[list[slice]], shape: torch.Size) -> list[list[slice]]:
    """Normalize slices so each has concrete start/stop (no None)."""
    result = []
    for d, dim_slices in enumerate(slices):
        normalized = []
        for s in dim_slices:
            start, stop, step = s.indices(shape[d])
            assert step == 1, f"GetSliceOp only supports step=1, got step={step} in dim {d}"
            normalized.append(slice(start, stop))
        result.append(normalized)
    return result


def _output_size(dim_slices: list[slice]) -> int:
    """Total output size for a list of slices along one dimension."""
    return sum(s.stop - s.start for s in dim_slices)


def _is_full(dim_slices: list[slice], dim_size: int) -> bool:
    """Check if slices cover the full dimension."""
    return len(dim_slices) == 1 and dim_slices[0].start == 0 and dim_slices[0].stop == dim_size


class GetSliceOp(LinearOp):
    """Extract sliced regions from a tensor.

    Args:
        input_shape: Shape of the input tensor.
        slices: Per-dimension list of slices. ``len(slices) == len(input_shape)``.
    """

    def __init__(self, input_shape: torch.Size, slices: list[list[slice]]):
        assert len(input_shape) == len(slices), \
            f"len(input_shape)={len(input_shape)} != len(slices)={len(slices)}"
        self.slices = _normalize_slices(slices, input_shape)
        output_shape = torch.Size(_output_size(self.slices[d]) for d in range(len(input_shape)))
        super().__init__(input_shape, output_shape,
                         flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for d, dim_slices in enumerate(self.slices):
            if _is_full(dim_slices, self.input_shape[d]):
                continue
            if len(dim_slices) == 1:
                s = dim_slices[0]
                x = x.narrow(d, s.start, s.stop - s.start)
            else:
                parts = [x.narrow(d, s.start, s.stop - s.start) for s in dim_slices]
                x = torch.cat(parts, dim=d)
        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        for d in range(len(self.slices)):
            dim_slices = self.slices[d]
            if _is_full(dim_slices, self.input_shape[d]):
                continue
            sizes = [s.stop - s.start for s in dim_slices]
            parts = grad.split(sizes, dim=d) if len(dim_slices) > 1 else [grad]
            shape = list(grad.shape)
            shape[d] = self.input_shape[d]
            result = torch.zeros(shape, dtype=grad.dtype, device=grad.device)
            for s, part in zip(dim_slices, parts):
                result.narrow(d, s.start, s.stop - s.start).add_(part)
            grad = result
        return grad

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        batch_ndim = grad.dim() - len(self.output_shape)
        for d in range(len(self.slices)):
            dim_slices = self.slices[d]
            bd = batch_ndim + d
            if _is_full(dim_slices, self.input_shape[d]):
                continue
            sizes = [s.stop - s.start for s in dim_slices]
            parts = grad.split(sizes, dim=bd) if len(dim_slices) > 1 else [grad]
            shape = list(grad.shape)
            shape[bd] = self.input_shape[d]
            result = torch.zeros(shape, dtype=grad.dtype, device=grad.device)
            for s, part in zip(dim_slices, parts):
                result.narrow(bd, s.start, s.stop - s.start).add_(part)
            grad = result
        return grad

    def __matmul__(self, other):
        """Fuse GetSliceOp @ GetSliceOp or GetSliceOp @ EinsumOp."""
        if isinstance(other, GetSliceOp):
            # Compose slices: apply self's slices to other's slices
            assert self.input_shape == other.output_shape
            new_slices = _compose_get_slices(self.slices, other.slices)
            return GetSliceOp(other.input_shape, new_slices)

        from boundlab.linearop._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.input_shape == other.output_shape
            # Check if all non-trivial slicing dims are dot/batch dims
            mul_slice_dims = []
            for d, dim_slices in enumerate(self.slices):
                if _is_full(dim_slices, self.input_shape[d]):
                    continue
                tensor_dim = other.output_dims[d]
                if tensor_dim in other.mul_dims:
                    mul_slice_dims.append(d)

            return _apply_getslice_einsum(self, other, mul_slice_dims)
        return NotImplemented

    def __rmatmul__(self, other):
        return super().__rmatmul__(other)

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
    """Embed input into zeros at specified slice positions.

    Args:
        output_shape: Shape of the output tensor (zeros template).
        slices: Per-dimension list of slices. ``len(output_shape) == len(slices)``.
    """

    def __init__(self, output_shape: torch.Size, slices: list[list[slice]]):
        assert len(output_shape) == len(slices), \
            f"len(output_shape)={len(output_shape)} != len(slices)={len(slices)}"
        self.slices = _normalize_slices(slices, output_shape)
        input_shape = torch.Size(_output_size(self.slices[d]) for d in range(len(output_shape)))
        super().__init__(input_shape, output_shape,
                         flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(self.output_shape, dtype=x.dtype, device=x.device)
        _scatter_slices(result, x, self.slices)
        return result

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return _gather_slices(grad, self.slices)

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[len(self.input_shape):]
        result = torch.zeros(*self.output_shape, *batch, dtype=x.dtype, device=x.device)
        _scatter_slices(result, x, self.slices)
        return result

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        batch_ndim = grad.dim() - len(self.output_shape)
        return _gather_slices_batched(grad, self.slices, batch_ndim)

    def __rmatmul__(self, other):
        """Fuse EinsumOp @ SetSliceOp."""
        from boundlab.linearop._einsum import EinsumOp
        if isinstance(other, EinsumOp) and self.output_shape == other.input_shape:
            # Check if all non-trivial slicing dims are dot/batch dims
            mul_slice_dims = []
            for d, dim_slices in enumerate(self.slices):
                if _is_full(dim_slices, self.output_shape[d]):
                    continue
                tensor_dim = other.input_dims[d]
                if tensor_dim in other.mul_dims:
                    mul_slice_dims.append(d)

            return _apply_einsum_setslice(other, self, mul_slice_dims)
        return super().__rmatmul__(other)

    def __matmul__(self, other):
        """Fuse SetSliceOp @ SetSliceOp."""
        if isinstance(other, SetSliceOp):
            assert self.input_shape == other.output_shape
            new_slices = _compose_set_slices(self.slices, other.slices)
            return SetSliceOp(self.output_shape, new_slices)
        return NotImplemented

    def __str__(self):
        parts = []
        for dim_slices in self.slices:
            if len(dim_slices) == 1:
                s = dim_slices[0]
                parts.append(f"{s.start}:{s.stop}")
            else:
                parts.append("[" + ",".join(f"{s.start}:{s.stop}" for s in dim_slices) + "]")
        return f"<setslice {','.join(parts)}>"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gather_slices(x: torch.Tensor, slices: list[list[slice]]) -> torch.Tensor:
    """Gather (forward of GetSliceOp)."""
    for d, dim_slices in enumerate(slices):
        if _is_full(dim_slices, x.shape[d]):
            continue
        if len(dim_slices) == 1:
            s = dim_slices[0]
            x = x.narrow(d, s.start, s.stop - s.start)
        else:
            parts = [x.narrow(d, s.start, s.stop - s.start) for s in dim_slices]
            x = torch.cat(parts, dim=d)
    return x


def _gather_slices_batched(grad: torch.Tensor, slices: list[list[slice]], batch_ndim: int) -> torch.Tensor:
    """Gather with leading batch dims."""
    for d, dim_slices in enumerate(slices):
        bd = batch_ndim + d
        if len(dim_slices) == 1 and dim_slices[0] == slice(0, grad.shape[bd]):
            continue
        if len(dim_slices) == 1:
            s = dim_slices[0]
            grad = grad.narrow(bd, s.start, s.stop - s.start)
        else:
            parts = [grad.narrow(bd, s.start, s.stop - s.start) for s in dim_slices]
            grad = torch.cat(parts, dim=bd)
    return grad


def _scatter_slices(result: torch.Tensor, x: torch.Tensor, slices: list[list[slice]]) -> None:
    """Scatter x into result at slice positions (in-place)."""
    _recursive_scatter(result, x, slices, 0, [], [])


def _recursive_scatter(result, x, slices, dim, result_indices, x_offsets):
    """Recursively scatter x into result across all slice combinations."""
    if dim == len(slices):
        r_idx = tuple(result_indices)
        x_idx = tuple(x_offsets)
        result[r_idx].copy_(x[x_idx])
        return

    dim_slices = slices[dim]
    x_pos = 0
    for s in dim_slices:
        length = s.stop - s.start
        _recursive_scatter(
            result, x, slices, dim + 1,
            result_indices + [slice(s.start, s.stop)],
            x_offsets + [slice(x_pos, x_pos + length)]
        )
        x_pos += length


def _compose_get_slices(outer_slices, inner_slices):
    """Compose GetSlice @ GetSlice: apply outer slices to inner's output."""
    result = []
    for d in range(len(outer_slices)):
        # inner_slices[d] maps positions in inner.input to inner.output
        # outer_slices[d] selects from inner.output
        inner = inner_slices[d]
        outer = outer_slices[d]
        new_dim_slices = _compose_dim_slices(outer, inner)
        result.append(new_dim_slices)
    return result


def _compose_dim_slices(outer: list[slice], inner: list[slice]) -> list[slice]:
    """Compose slices along one dimension.

    ``inner`` maps from the original tensor to an intermediate.
    ``outer`` selects from the intermediate.
    Result maps from the original tensor directly.
    """
    # Build a mapping from intermediate positions to original positions
    # inner creates segments: inner[0] -> positions 0..len0, inner[1] -> len0..len0+len1, etc.
    result = []
    for o_slice in outer:
        # Find which inner slices cover positions o_slice.start..o_slice.stop
        pos = 0
        remaining_start = o_slice.start
        remaining_stop = o_slice.stop
        for i_slice in inner:
            i_len = i_slice.stop - i_slice.start
            seg_start = pos
            seg_stop = pos + i_len
            # Intersection of [remaining_start, remaining_stop) with [seg_start, seg_stop)
            inter_start = max(remaining_start, seg_start)
            inter_stop = min(remaining_stop, seg_stop)
            if inter_start < inter_stop:
                # Map back to original coordinates
                orig_start = i_slice.start + (inter_start - seg_start)
                orig_stop = i_slice.start + (inter_stop - seg_start)
                result.append(slice(orig_start, orig_stop))
            pos += i_len
    return _merge_adjacent_slices(result)


def _compose_set_slices(outer_slices, inner_slices):
    """Compose SetSlice @ SetSlice: embed inner's output into outer's output."""
    result = []
    for d in range(len(outer_slices)):
        outer = outer_slices[d]
        inner = inner_slices[d]
        new_dim_slices = _compose_dim_slices(inner, outer)
        result.append(new_dim_slices)
    return result


def _merge_adjacent_slices(slices: list[slice]) -> list[slice]:
    """Merge adjacent/overlapping slices."""
    if not slices:
        return [slice(0, 0)]
    result = [slices[0]]
    for s in slices[1:]:
        if s.start <= result[-1].stop:
            result[-1] = slice(result[-1].start, max(result[-1].stop, s.stop))
        else:
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Fusion with EinsumOp
# ---------------------------------------------------------------------------


def _slice_tensor_along_dims(tensor, dims_map, slices_map):
    """Slice tensor along multiple dims. dims_map[output_d] = tensor_dim; slices_map[output_d] = dim_slices."""
    for output_d, tensor_dim in dims_map.items():
        dim_slices = slices_map[output_d]
        if len(dim_slices) == 1:
            s = dim_slices[0]
            tensor = tensor.narrow(tensor_dim, s.start, s.stop - s.start)
        else:
            parts = [tensor.narrow(tensor_dim, s.start, s.stop - s.start) for s in dim_slices]
            tensor = torch.cat(parts, dim=tensor_dim)
    return tensor


def _apply_getslice_einsum(gs: GetSliceOp, einsum, mul_dims: list[int]):
    """Fuse/swap GetSliceOp @ EinsumOp.

    Slices the tensor on all non-trivial dims. For mul dims, also adds a
    GetSliceOp on the input side (swap). For dot/batch dims, no input op needed (fuse).
    """
    from boundlab.linearop._einsum import EinsumOp

    dims_map = {}
    slices_map = {}
    for d, dim_slices in enumerate(gs.slices):
        if not _is_full(dim_slices, gs.input_shape[d]):
            dims_map[d] = einsum.output_dims[d]
            slices_map[d] = dim_slices

    tensor = _slice_tensor_along_dims(einsum.tensor, dims_map, slices_map)
    new_einsum = EinsumOp(tensor, einsum.input_dims, einsum.output_dims,
                          name=merge_name(gs, "@", einsum))
    assert new_einsum.output_shape == gs.output_shape, \
        f"_apply_getslice_einsum: output_shape {new_einsum.output_shape} != {gs.output_shape}"

    if not mul_dims:
        assert new_einsum.input_shape == einsum.input_shape, \
            f"_apply_getslice_einsum: input_shape {new_einsum.input_shape} != {einsum.input_shape}"
        return new_einsum

    # Build input-side slices for mul dims
    input_side_slices = [[slice(0, einsum.input_shape[d])] for d in range(len(einsum.input_shape))]
    for d in mul_dims:
        tensor_dim = einsum.output_dims[d]
        input_d = einsum.input_dims.index(tensor_dim)
        input_side_slices[input_d] = gs.slices[d]

    needs_input_slice = any(
        not _is_full(input_side_slices[d], einsum.input_shape[d])
        for d in range(len(einsum.input_shape))
    )
    if needs_input_slice:
        input_gs = GetSliceOp(einsum.input_shape, input_side_slices)
        return ComposedOp(new_einsum, input_gs)
    return new_einsum


def _apply_einsum_setslice(einsum, ss: SetSliceOp, mul_dims: list[int]):
    """Fuse/swap EinsumOp @ SetSliceOp.

    Slices the tensor on all non-trivial dims. For mul dims, also adds a
    SetSliceOp on the output side (swap). For dot/batch dims, no output op needed (fuse).
    """
    from boundlab.linearop._einsum import EinsumOp

    dims_map = {}
    slices_map = {}
    for d, dim_slices in enumerate(ss.slices):
        if not _is_full(dim_slices, ss.output_shape[d]):
            dims_map[d] = einsum.input_dims[d]
            slices_map[d] = dim_slices

    tensor = _slice_tensor_along_dims(einsum.tensor, dims_map, slices_map)
    new_einsum = EinsumOp(tensor, einsum.input_dims, einsum.output_dims,
                          name=merge_name(einsum, "@", ss))
    assert new_einsum.input_shape == ss.input_shape, \
        f"_apply_einsum_setslice: input_shape {new_einsum.input_shape} != {ss.input_shape}"

    if not mul_dims:
        assert new_einsum.output_shape == einsum.output_shape, \
            f"_apply_einsum_setslice: output_shape {new_einsum.output_shape} != {einsum.output_shape}"
        return new_einsum

    # Build output-side slices for mul dims
    output_side_slices = [[slice(0, einsum.output_shape[d])] for d in range(len(einsum.output_shape))]
    for d in mul_dims:
        tensor_dim = einsum.input_dims[d]
        output_d = einsum.output_dims.index(tensor_dim)
        output_side_slices[output_d] = ss.slices[d]

    needs_output_slice = any(
        not _is_full(output_side_slices[d], einsum.output_shape[d])
        for d in range(len(einsum.output_shape))
    )
    if needs_output_slice:
        output_ss = SetSliceOp(einsum.output_shape, output_side_slices)
        return ComposedOp(output_ss, new_einsum)
    return new_einsum
