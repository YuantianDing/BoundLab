"""Indexing LinearOp implementations for bound propagation.

This module provides LinearOps for various indexing and slicing operations.
The key distinction is:

- **Slice-based** (`GetSliceOp`, `SetSliceOp`): Use Python slice/int indices.
  These are the generalized versions that subsume `NarrowOp`, `SelectOp`,
  `GetItemOp`, and `PadOp`.

- **Index-tensor-based** (`GetIndicesOp`, `SetIndicesOp`): Use tensor indices
  for advanced indexing where each element position is specified by tensors.

- **Gather/Scatter** (`GatherOp`, `ScatterOp`): Dimension-specific operations
  that gather or scatter along a single dimension using index tensors.
"""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags
from boundlab.utils import merge_name


def _meta_output_shape(fn, input_shape: torch.Size) -> torch.Size:
    """Infer an operator output shape without materializing data.

    Args:
        fn: A shape-preserving callable that accepts a tensor.
        input_shape: Shape of the hypothetical input tensor.

    Returns:
        The output shape obtained by running ``fn`` on a meta-device tensor.
    """
    return fn(torch.empty(input_shape, device="meta")).shape


# ---------------------------------------------------------------------------
# Gather / Scatter operations
# ---------------------------------------------------------------------------


class GatherOp(LinearOp):
    """A LinearOp that implements ``torch.gather`` along a specified dimension.

    Forward: ``output[i][j][k] = input[i][index[i][j][k]][k]`` (for dim=1)
    Backward: Scatters gradient back using ``scatter_add``.
    """

    def __init__(self, input_shape: torch.Size, dim: int, index: torch.Tensor):
        self.dim = dim
        self.index = index
        output_shape = torch.Size(index.shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if x has extra batch dimensions (e.g., when called via vmap)
        if x.shape != self.input_shape:
            return self.vforward(x)
        return torch.gather(x, self.dim, self.index)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        # Check if grad has extra batch dimensions
        if grad.shape != self.output_shape:
            return self.vbackward(grad)
        result = torch.zeros(self.input_shape, dtype=grad.dtype, device=grad.device)
        result.scatter_add_(self.dim, self.index, grad)
        return result

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*input_shape, *batch_dims)
        batch_dims = x.shape[len(self.input_shape):]
        # Expand index to match batch dims
        index = self.index
        for _ in batch_dims:
            index = index.unsqueeze(-1)
        index = index.expand(*self.index.shape, *batch_dims)
        return torch.gather(x, self.dim, index)

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        # grad: (*batch_dims, *output_shape)
        batch_dims = grad.shape[:-len(self.output_shape)]
        batch_ndim = len(batch_dims)
        # Expand index to match batch dims
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
    """A LinearOp that implements ``torch.scatter`` along a specified dimension.

    Forward: Creates zeros of output_shape, then scatters input values at index positions.
    Backward: Gathers gradient from the scattered positions.
    """

    def __init__(self, input_shape: torch.Size, dim: int, index: torch.Tensor, output_shape: torch.Size):
        self.dim = dim
        self.index = index
        assert index.shape == input_shape, f"Index shape {index.shape} must match input shape {input_shape}"
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if x has extra batch dimensions
        if x.shape != self.input_shape:
            return self.vforward(x)
        result = torch.zeros(self.output_shape, dtype=x.dtype, device=x.device)
        result.scatter_(self.dim, self.index, x)
        return result

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        # Check if grad has extra batch dimensions
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
# Advanced indexing with index tensors
# ---------------------------------------------------------------------------


class GetIndicesOp(LinearOp):
    r"""Advanced indexing with a tuple of index tensors.

    Given ``indices = (i_0, i_1, ..., i_{d-1})``, forward evaluation computes:

    .. math::

       y = x[\text{indices}]

    where each index tensor has shape ``output_shape``.
    The transpose (backward) operation writes each entry of ``grad`` back to
    its indexed position in an all-zero tensor, with accumulation for repeated
    indices.

    Args:
        indices: Tuple of integer index tensors, one per input dimension.
        input_shape: Shape of the source tensor ``x``.
        output_shape: Shape of the indexed output tensor ``y``.

    Notes:
        If an index appears multiple times, gradients are summed at that
        position (``accumulate=True`` semantics).
    """

    def __init__(self, indices: tuple[torch.Tensor, ...], input_shape: torch.Size, output_shape: torch.Size):
        self.indices = indices
        assert isinstance(indices, tuple) and len(indices) == len(input_shape), \
            "Indices must be a tuple of the same length as input_shape."
        for idx in indices:
            assert idx.shape == output_shape, \
                f"Each index tensor must have shape {output_shape}, got {idx.shape}"
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if x has extra batch dimensions
        if x.shape != self.input_shape:
            return self.vforward(x)
        return x[self.indices]

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        # Check if grad has extra batch dimensions
        if grad.shape != self.output_shape:
            return self.vbackward(grad)
        result = torch.zeros(self.input_shape, dtype=grad.dtype, device=grad.device)
        # Use index_put_ with accumulate=True for correct gradient with repeated indices
        result.index_put_(self.indices, grad, accumulate=True)
        return result

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*input_shape, *batch)
        batch = x.shape[len(self.input_shape):]
        if not batch:
            return x[self.indices]
        # Expand indices to include batch dims
        expanded_indices = []
        for idx in self.indices:
            for _ in batch:
                idx = idx.unsqueeze(-1)
            expanded_indices.append(idx.expand(*self.output_shape, *batch))
        return x[tuple(expanded_indices)]

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        # grad: (*batch, *output_shape)
        batch = grad.shape[:-len(self.output_shape)]
        batch_ndim = len(batch)
        result = torch.zeros(*batch, *self.input_shape, dtype=grad.dtype, device=grad.device)
        expanded_indices = []
        for idx in self.indices:
            for _ in batch:
                idx = idx.unsqueeze(0)
            expanded_indices.append(idx.expand(*batch, *self.output_shape))
        # Prepend batch indices
        batch_indices = [
            torch.arange(b, device=grad.device).reshape(
                *([1] * i), b, *([1] * (batch_ndim - i - 1 + len(self.output_shape)))
            ).expand(*batch, *self.output_shape)
            for i, b in enumerate(batch)
        ]
        all_indices = tuple(batch_indices) + tuple(expanded_indices)
        result.index_put_(all_indices, grad, accumulate=True)
        return result

    def __str__(self):
        return f"<get_indices {tuple(self.output_shape)}>"


class SetIndicesOp(LinearOp):
    """Scatter values to advanced index positions in a zero-initialized tensor.

    Forward creates ``result = zeros(output_shape)`` and assigns:
    ``result[indices] = input``.
    Backward gathers gradients at the same index positions.

    This operator is the transpose/adjoint counterpart of
    :class:`GetIndicesOp`.
    """

    def __init__(self, indices: tuple[torch.Tensor, ...], input_shape: torch.Size, output_shape: torch.Size):
        self.indices = indices
        assert isinstance(indices, tuple) and len(indices) == len(output_shape), \
            "Indices must be a tuple of the same length as output_shape."
        for idx in indices:
            assert idx.shape == input_shape, \
                f"Each index tensor must have shape {input_shape}, got {idx.shape}"
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if x has extra batch dimensions
        if x.shape != self.input_shape:
            return self.vforward(x)
        result = torch.zeros(self.output_shape, dtype=x.dtype, device=x.device)
        result[self.indices] = x
        return result

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        # Check if grad has extra batch dimensions
        if grad.shape != self.output_shape:
            return self.vbackward(grad)
        return grad[self.indices]

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[len(self.input_shape):]
        if not batch:
            return self.forward(x)
        expanded_indices = []
        for idx in self.indices:
            for _ in batch:
                idx = idx.unsqueeze(-1)
            expanded_indices.append(idx.expand(*self.input_shape, *batch))
        result = torch.zeros(*self.output_shape, *batch, dtype=x.dtype, device=x.device)
        result[tuple(expanded_indices)] = x
        return result

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        batch = grad.shape[:-len(self.output_shape)]
        expanded_indices = []
        for idx in self.indices:
            for _ in batch:
                idx = idx.unsqueeze(0)
            expanded_indices.append(idx.expand(*batch, *self.input_shape))
        batch_indices = [
            torch.arange(b, device=grad.device).reshape(
                *([1] * i), b, *([1] * (len(batch) - i - 1 + len(self.input_shape)))
            ).expand(*batch, *self.input_shape)
            for i, b in enumerate(batch)
        ]
        all_indices = tuple(batch_indices) + tuple(expanded_indices)
        return grad[all_indices]

    def __str__(self):
        return f"<set_indices {list(self.input_shape)} -> {list(self.output_shape)}>"


# ---------------------------------------------------------------------------
# Slice-based indexing (basic indexing with int/slice)
# ---------------------------------------------------------------------------


def _normalize_basic_indices(indices, ndim: int):
    """Expand ``indices`` (as passed to ``x[indices]``) into a length-``ndim``
    list of ``slice`` / ``int`` entries.

    Returns ``None`` when the indices include ``None`` (newaxis), tensor
    advanced indices, or anything else not representable per-axis as a basic
    slice/int — the fusion paths below bail out in that case.
    """
    if not isinstance(indices, tuple):
        indices = (indices,)
    normalized: list = []
    idx_pos = 0
    saw_ellipsis = False
    for _ in range(ndim):
        if idx_pos < len(indices):
            idx = indices[idx_pos]
            if idx is Ellipsis:
                if saw_ellipsis:
                    return None
                saw_ellipsis = True
                remaining = ndim - (len(indices) - 1 - idx_pos) - len(normalized)
                for _ in range(remaining):
                    normalized.append(slice(None))
                idx_pos += 1
                continue
            if isinstance(idx, slice) or isinstance(idx, int):
                normalized.append(idx)
                idx_pos += 1
            else:
                return None
        else:
            normalized.append(slice(None))
    if idx_pos != len(indices):
        return None
    return normalized


class GetSliceOp(LinearOp):
    """Basic slicing via ``x[indices]`` where indices contains int/slice/None/Ellipsis.

    This is a generalization that subsumes:
    - NarrowOp: ``x.narrow(dim, start, length)`` → ``GetSliceOp`` with a slice at dim
    - SelectOp: ``x.select(dim, index)`` → ``GetSliceOp`` with an int at dim
    - GetItemOp: ``x[indices]`` → ``GetSliceOp`` directly

    Forward: ``output = input[indices]``
    Backward: Embeds gradient into zeros at the sliced positions.
    """

    def __init__(self, input_shape: torch.Size, indices):
        self.indices = indices
        output_shape = _meta_output_shape(lambda x: x[indices], input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)
        # Pre-compute backward info for functional (vmap-compatible) backward
        self._backward_info = self._compute_backward_info(input_shape, indices, output_shape)

    def _compute_backward_info(self, input_shape, indices, output_shape):  # noqa: ARG002
        """Pre-compute info needed for functional backward using F.pad."""
        # Normalize indices to a tuple
        if not isinstance(indices, tuple):
            indices = (indices,)

        # Check if we can use F.pad (pure slices, no integer indices that reduce dims)
        # and compute the pad_spec
        ndim = len(input_shape)
        normalized = []
        int_indices = []  # (dim, index) pairs for integer indices

        idx_pos = 0
        for i in range(ndim):
            if idx_pos < len(indices):
                idx = indices[idx_pos]
                if idx is Ellipsis:
                    # Ellipsis expands to fill remaining dims
                    remaining = ndim - len(indices) + 1
                    for _ in range(remaining):
                        normalized.append(slice(None))
                    idx_pos += 1
                    continue
                elif isinstance(idx, int):
                    int_indices.append((i, idx))
                    normalized.append(idx)
                elif isinstance(idx, slice):
                    normalized.append(idx)
                elif idx is None:
                    # newaxis - adds dimension, handle separately
                    normalized.append(idx)
                else:
                    normalized.append(idx)
                idx_pos += 1
            else:
                normalized.append(slice(None))

        # If there are integer indices, we need to unsqueeze before padding
        # Compute pad_spec from normalized indices
        pad_spec = []
        grad_dim = 0  # Current dimension in grad
        for d in reversed(range(ndim)):
            idx = normalized[d] if d < len(normalized) else slice(None)
            if isinstance(idx, int):
                # Integer index - this dim is removed in output, need to unsqueeze
                pad_before = idx if idx >= 0 else input_shape[d] + idx
                pad_after = input_shape[d] - pad_before - 1
                pad_spec.extend([pad_before, pad_after])
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(input_shape[d])
                if step != 1:
                    return None  # Can't use F.pad for step != 1
                pad_before = start
                pad_after = input_shape[d] - stop
                pad_spec.extend([pad_before, pad_after])
                grad_dim += 1
            else:
                pad_spec.extend([0, 0])
                grad_dim += 1

        return {
            'pad_spec': pad_spec,
            'int_indices': int_indices,
            'normalized': normalized,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.indices]

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward using F.pad for vmap compatibility."""
        import torch.nn.functional as F

        info = self._backward_info
        if info is None:
            # Fallback for complex indexing (step != 1, etc.)
            result = torch.zeros(self.input_shape, dtype=grad.dtype, device=grad.device)
            result[self.indices] = grad
            return result

        # First, unsqueeze for any integer indices (in reverse order to maintain positions)
        result = grad
        for dim, _ in reversed(info['int_indices']):
            result = result.unsqueeze(dim)

        # Then apply padding
        if any(p != 0 for p in info['pad_spec']):
            result = F.pad(result, info['pad_spec'])

        return result

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*input_shape, *batch)
        # Apply slicing to the leading dims, preserve trailing batch dims
        return x[self.indices]

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward with leading batch dimensions."""
        import torch.nn.functional as F

        info = self._backward_info
        batch_ndim = len(grad.shape) - len(self.output_shape)

        if info is None:
            batch = grad.shape[:batch_ndim]
            result = torch.zeros(*batch, *self.input_shape, dtype=grad.dtype, device=grad.device)
            batch_slices = tuple(slice(None) for _ in range(batch_ndim))
            result[batch_slices + self.indices] = grad
            return result

        result = grad
        for dim, _ in reversed(info['int_indices']):
            result = result.unsqueeze(batch_ndim + dim)

        if any(p != 0 for p in info['pad_spec']):
            result = F.pad(result, info['pad_spec'])

        return result

    def __str__(self):
        return f"<getslice {_format_indices(self.indices)}>"

    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        """Fuse ``GetSliceOp @ EinsumOp`` by slicing the einsum tensor along
        output-side tensor dims.

        Bails out (``NotImplemented``) when any sliced output axis is a
        mul_dim (shared input/output tensor dim): slicing such a dim would
        implicitly reshape the input as well, which requires
        ``remove_conditions`` to express correctly.
        """
        from boundlab.linearop._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            if self.input_shape != other.output_shape:
                return NotImplemented
            normalized = _normalize_basic_indices(self.indices, len(other.output_shape))
            if normalized is None:
                return NotImplemented
            for out_axis, idx in enumerate(normalized):
                tensor_dim = other.output_dims[out_axis]
                if tensor_dim in other.mul_dims:
                    if isinstance(idx, slice) and idx == slice(None):
                        continue
                    return NotImplemented

            tensor_slices: list = [slice(None)] * other.tensor.dim()
            dropped: list[int] = []
            for out_axis, idx in enumerate(normalized):
                tensor_dim = other.output_dims[out_axis]
                if isinstance(idx, int):
                    tensor_slices[tensor_dim] = idx
                    dropped.append(tensor_dim)
                else:
                    tensor_slices[tensor_dim] = idx
            new_tensor = other.tensor[tuple(tensor_slices)]

            def adj(d: int) -> int:
                return d - sum(1 for dd in dropped if dd < d)

            new_output_dims = [
                adj(other.output_dims[i])
                for i, idx in enumerate(normalized)
                if not isinstance(idx, int)
            ]
            new_input_dims = [adj(d) for d in other.input_dims]
            return EinsumOp(new_tensor, new_input_dims, new_output_dims,
                            name=merge_name(self, "@", other))
        return other.__rmatmul__(self)
    
    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        return super().__rmatmul__(other)


class SetSliceOp(LinearOp):
    """Embed input into zeros at specified slice positions.

    This is the adjoint/transpose of GetSliceOp and generalizes PadOp.

    Forward: Creates zeros of output_shape, sets ``result[indices] = input``.
    Backward: Extracts gradient at the sliced positions.
    """

    def __init__(self, indices, input_shape: torch.Size, output_shape: torch.Size):
        self.indices = indices
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if x has extra batch dimensions (e.g., when called via vmap)
        if x.shape != self.input_shape:
            return self.vforward(x)
        result = torch.zeros(self.output_shape, dtype=x.dtype, device=x.device)
        result[self.indices] = x
        return result

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        # Check if grad has extra batch dimensions
        if grad.shape != self.output_shape:
            return self.vbackward(grad)
        return grad[self.indices]

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[len(self.input_shape):]
        if not batch:
            return self.forward(x)
        result = torch.zeros(*self.output_shape, *batch, dtype=x.dtype, device=x.device)
        result[self.indices] = x
        return result

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        batch = grad.shape[:-len(self.output_shape)] if self.output_shape else grad.shape
        if not batch:
            return self.backward(grad)
        batch_ndim = len(batch)
        batch_slices = tuple(slice(None) for _ in range(batch_ndim))
        if isinstance(self.indices, tuple):
            full_indices = batch_slices + self.indices
        else:
            full_indices = batch_slices + (self.indices,)
        return grad[full_indices]

    def __str__(self):
        return f"<setslice {_format_indices(self.indices)}>"

    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        """Fuse ``EinsumOp @ SetSliceOp`` by slicing the einsum tensor along
        input-side tensor dims.

        ``SetSlice`` embeds its (small) input into zeros at indexed positions
        of an (einsum-input-shaped) tensor; zero entries contribute nothing to
        the einsum, so the composition equals an einsum whose tensor is sliced
        to the indexed region.  Bails out when a sliced input axis is a
        mul_dim (would also change the output shape).
        """
        from boundlab.linearop._einsum import EinsumOp
        if isinstance(other, EinsumOp) and self.output_shape == other.input_shape:
            normalized = _normalize_basic_indices(self.indices, len(other.input_shape))
            fusable = normalized is not None
            if fusable:
                for in_axis, idx in enumerate(normalized):
                    tensor_dim = other.input_dims[in_axis]
                    if tensor_dim in other.mul_dims and not (
                        isinstance(idx, slice) and idx == slice(None)
                    ):
                        fusable = False
                        break
            if not fusable:
                return super().__rmatmul__(other)

            tensor_slices: list = [slice(None)] * other.tensor.dim()
            dropped: list[int] = []
            for in_axis, idx in enumerate(normalized):
                tensor_dim = other.input_dims[in_axis]
                if isinstance(idx, int):
                    tensor_slices[tensor_dim] = idx
                    dropped.append(tensor_dim)
                else:
                    tensor_slices[tensor_dim] = idx
            new_tensor = other.tensor[tuple(tensor_slices)]

            def adj(d: int) -> int:
                return d - sum(1 for dd in dropped if dd < d)

            new_input_dims = [
                adj(other.input_dims[i])
                for i, idx in enumerate(normalized)
                if not isinstance(idx, int)
            ]
            new_output_dims = [adj(d) for d in other.output_dims]
            return EinsumOp(new_tensor, new_input_dims, new_output_dims,
                            name=merge_name(other, "@", self))
        return super().__rmatmul__(other)


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
    """Create slice indices for embedding input into a padded output.

    The pad_spec follows PyTorch's F.pad convention: [left, right, top, bottom, ...]
    applied from the last dimension backwards.
    """
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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
