"""Indexing LinearOp implementations for bound propagation.

This module provides:
- GatherOp, ScatterOp: Dimension-specific gather/scatter operations.
- Re-exports from _slicing and _indexing for backward compatibility.
- Convenience constructors for common slice patterns.
"""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags

# Re-export new ops for backward compatibility
from boundlab.linearop._slicing import GetSliceOp, SetSliceOp
from boundlab.linearop._indexing import GetIndicesOp, SetIndicesOp


# ---------------------------------------------------------------------------
# Gather / Scatter operations
# ---------------------------------------------------------------------------


class GatherOp(LinearOp):
    """A LinearOp that implements ``torch.gather`` along a specified dimension."""

    def __init__(self, input_shape: torch.Size, dim: int, index: torch.Tensor):
        self.dim = dim
        self.index = index
        output_shape = torch.Size(index.shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape != self.input_shape:
            return self.vforward(x)
        return torch.gather(x, self.dim, self.index)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.shape != self.output_shape:
            return self.vbackward(grad)
        result = torch.zeros(self.input_shape, dtype=grad.dtype, device=grad.device)
        result.scatter_add_(self.dim, self.index, grad)
        return result

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        batch_dims = x.shape[len(self.input_shape):]
        index = self.index
        for _ in batch_dims:
            index = index.unsqueeze(-1)
        index = index.expand(*self.index.shape, *batch_dims)
        return torch.gather(x, self.dim, index)

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        batch_dims = grad.shape[:-len(self.output_shape)]
        batch_ndim = len(batch_dims)
        index = self.index
        for _ in batch_dims:
            index = index.unsqueeze(0)
        index = index.expand(*batch_dims, *self.index.shape)
        result = torch.zeros(*batch_dims, *self.input_shape, dtype=grad.dtype, device=grad.device)
        result.scatter_add_(batch_ndim + self.dim, index, grad)
        return result

    def __str__(self):
        return f"<gather dim={self.dim} index.shape={list(self.index.shape)}>"


class ScatterOp(LinearOp):
    """A LinearOp that implements ``torch.scatter`` along a specified dimension."""

    def __init__(self, input_shape: torch.Size, dim: int, index: torch.Tensor, output_shape: torch.Size):
        self.dim = dim
        self.index = index
        assert index.shape == input_shape, f"Index shape {index.shape} must match input shape {input_shape}"
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape != self.input_shape:
            return self.vforward(x)
        result = torch.zeros(self.output_shape, dtype=x.dtype, device=x.device)
        result.scatter_(self.dim, self.index, x)
        return result

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.shape != self.output_shape:
            return self.vbackward(grad)
        return torch.gather(grad, self.dim, self.index)

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        batch_dims = x.shape[len(self.input_shape):]
        index = self.index
        for _ in batch_dims:
            index = index.unsqueeze(-1)
        index = index.expand(*self.index.shape, *batch_dims)
        result = torch.zeros(*self.output_shape, *batch_dims, dtype=x.dtype, device=x.device)
        result.scatter_(self.dim, index, x)
        return result

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        batch_dims = grad.shape[:-len(self.output_shape)]
        batch_ndim = len(batch_dims)
        index = self.index
        for _ in batch_dims:
            index = index.unsqueeze(0)
        index = index.expand(*batch_dims, *self.index.shape)
        return torch.gather(grad, batch_ndim + self.dim, index)

    def __str__(self):
        return f"<scatter dim={self.dim} index.shape={list(self.index.shape)}>"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def narrow_indices(ndim: int, dim: int, start: int, length: int) -> tuple:
    """Create slice indices equivalent to ``tensor.narrow(dim, start, length)``."""
    indices = [slice(None)] * ndim
    indices[dim] = slice(start, start + length)
    return tuple(indices)


def select_indices(ndim: int, dim: int, index: int) -> tuple:
    """Create slice indices equivalent to ``tensor.select(dim, index)``."""
    indices = [slice(None)] * ndim
    indices[dim] = index
    return tuple(indices)


def pad_indices(input_shape: torch.Size, pad_spec: list[int]) -> tuple:
    """Create slice indices for embedding input into a padded output."""
    ndim = len(input_shape)
    indices = []
    for d in range(ndim):
        d_rev = ndim - 1 - d
        if 2 * d_rev + 1 < len(pad_spec):
            pad_before = pad_spec[2 * d_rev]
            indices.append(slice(pad_before, pad_before + input_shape[d]))
        else:
            indices.append(slice(None))
    return tuple(indices)


def pad_output_shape(input_shape: torch.Size, pad_spec: list[int]) -> torch.Size:
    """Compute output shape after padding."""
    ndim = len(input_shape)
    output = list(input_shape)
    for d in range(ndim):
        d_rev = ndim - 1 - d
        if 2 * d_rev + 1 < len(pad_spec):
            output[d] += pad_spec[2 * d_rev] + pad_spec[2 * d_rev + 1]
    return torch.Size(output)


def _format_indices(indices) -> str:
    """Format indices for string representation."""
    if not isinstance(indices, tuple):
        indices = (indices,)
    parts = []
    for idx in indices:
        if isinstance(idx, slice):
            start = "" if idx.start is None else str(idx.start)
            stop = "" if idx.stop is None else str(idx.stop)
            step = "" if idx.step is None else f":{idx.step}"
            parts.append(f"{start}:{stop}{step}")
        elif idx is None:
            parts.append("None")
        elif idx is Ellipsis:
            parts.append("...")
        else:
            parts.append(str(idx))
    return ", ".join(parts)


def make_get_slices(input_shape: torch.Size, indices) -> list[list["slice"]]:
    """Convert arbitrary Python indices to the structured ``list[list[slice]]`` format.
    
    Integer indices are converted to length-1 slices (dim is NOT removed).
    For dimension removal, compose with SqueezeOp.
    """
    if not isinstance(indices, tuple):
        indices = (indices,)
    
    ndim = len(input_shape)
    normalized = []
    idx_pos = 0
    saw_ellipsis = False
    
    for _ in range(ndim):
        if idx_pos < len(indices):
            idx = indices[idx_pos]
            if idx is Ellipsis:
                if saw_ellipsis:
                    raise ValueError("Only one Ellipsis allowed")
                saw_ellipsis = True
                remaining = ndim - (len(indices) - 1 - idx_pos) - len(normalized)
                for _ in range(remaining):
                    normalized.append([slice(None)])
                idx_pos += 1
                continue
            if isinstance(idx, int):
                if idx < 0:
                    idx += input_shape[len(normalized)]
                normalized.append([slice(idx, idx + 1)])
            elif isinstance(idx, slice):
                s = idx
                dim_size = input_shape[len(normalized)]
                start, stop, step = s.indices(dim_size)
                if step == 1:
                    normalized.append([slice(start, stop)])
                else:
                    # Convert step slices to multiple contiguous slices
                    positions = list(range(start, stop, step))
                    # Merge consecutive positions into contiguous slices
                    dim_slices = []
                    i = 0
                    while i < len(positions):
                        run_start = positions[i]
                        while i + 1 < len(positions) and positions[i + 1] == positions[i] + 1:
                            i += 1
                        dim_slices.append(slice(run_start, positions[i] + 1))
                        i += 1
                    normalized.append(dim_slices)
            else:
                raise ValueError(f"Unsupported index type: {type(idx)}")
            idx_pos += 1
        else:
            normalized.append([slice(None)])
    
    # Normalize slice(None) to concrete bounds
    result = []
    for d, dim_slices in enumerate(normalized):
        dim_result = []
        for s in dim_slices:
            if s == slice(None):
                dim_result.append(slice(0, input_shape[d]))
            else:
                dim_result.append(s)
        result.append(dim_result)
    return result


def make_set_slices(output_shape: torch.Size, indices) -> list[list["slice"]]:
    """Convert arbitrary Python indices to the structured format for SetSliceOp."""
    return make_get_slices(output_shape, indices)


def get_int_dims(indices) -> list[int]:
    """Return which dimensions use integer indices (should be squeezed)."""
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
