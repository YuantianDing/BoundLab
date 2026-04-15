"""Index-tensor-based LinearOp implementations.

``GetIndicesOp`` and ``SetIndicesOp`` index along a single dimension using
a tensor of indices, replacing that dimension with ``added_shape``.
"""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags, ComposedOp
from boundlab.utils import merge_name


class GetIndicesOp(LinearOp):
    """Gather elements along *dim* using an index tensor.

    output_shape = input_shape[:dim] + added_shape + input_shape[dim+1:]

    Args:
        input_shape: Shape of the source tensor.
        dim: Dimension along which to index.
        indices: Index tensor with shape ``added_shape``, values in ``[0, input_shape[dim])``.
        added_shape: Shape that replaces ``input_shape[dim]``.
    """

    def __init__(self, input_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size):
        self.dim = dim
        self.indices = indices
        self.added_shape = added_shape
        assert indices.shape == added_shape, \
            f"indices.shape={indices.shape} != added_shape={added_shape}"
        output_shape = torch.Size(
            list(input_shape[:dim]) + list(added_shape) + list(input_shape[dim + 1:])
        )
        super().__init__(input_shape, output_shape,
                         flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._gather(x, self.dim)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return self._scatter(grad, self.dim, self.input_shape[self.dim])

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        return self._gather(x, self.dim)

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        batch_ndim = grad.dim() - len(self.output_shape)
        return self._scatter(grad, batch_ndim + self.dim, self.input_shape[self.dim])

    def _gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Gather along dim, replacing it with added_shape."""
        n_added = len(self.added_shape)
        # Reshape indices for broadcasting: insert dims for all non-dim axes of x
        idx = self.indices
        # Add trailing dims for dims after 'dim' in x
        n_after = x.dim() - dim - 1
        for _ in range(n_after):
            idx = idx.unsqueeze(-1)
        # Add leading dims for dims before 'dim' in x
        for _ in range(dim):
            idx = idx.unsqueeze(0)
        # Now idx has shape: (1,)*dim + added_shape + (1,)*n_after
        # Expand to match x's shape except at the indexed dim (replaced by added_shape)
        expand_shape = list(x.shape[:dim]) + list(self.added_shape) + list(x.shape[dim + 1:])
        idx = idx.expand(expand_shape)
        # Flatten added_shape into a single dim for torch.gather
        # Reshape x: merge nothing (gather directly if added_shape is 1D)
        if n_added == 1:
            return torch.gather(x, dim, idx)
        else:
            # Flatten added_shape dims in idx
            flat_idx = idx.flatten(dim, dim + n_added - 1)
            # Insert extra dims in x to match
            result = torch.gather(x, dim, flat_idx)
            # Unflatten back to added_shape
            return result.unflatten(dim, self.added_shape)

    def _scatter(self, grad: torch.Tensor, dim: int, source_size: int) -> torch.Tensor:
        """Scatter gradients back along dim."""
        n_added = len(self.added_shape)
        # Flatten added_shape dims
        if n_added > 1:
            grad = grad.flatten(dim, dim + n_added - 1)

        idx = self.indices.reshape(-1)  # flatten added_shape
        # Reshape idx for broadcasting
        n_after = grad.dim() - dim - 1
        for _ in range(n_after):
            idx = idx.unsqueeze(-1)
        for _ in range(dim):
            idx = idx.unsqueeze(0)
        idx = idx.expand_as(grad)

        result_shape = list(grad.shape)
        result_shape[dim] = source_size
        result = torch.zeros(result_shape, dtype=grad.dtype, device=grad.device)
        result.scatter_add_(dim, idx, grad)
        return result

    def __matmul__(self, other):
        """Fuse GetIndicesOp @ EinsumOp."""
        from boundlab.linearop._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.input_shape == other.output_shape
            tensor_dim = other.output_dims[self.dim]
            if tensor_dim in other.mul_dims:
                # Mul dim: swap past einsum
                return _swap_getindices_einsum(self, other)
            else:
                # Dot/batch dim: fuse by indexing tensor
                return _fuse_getindices_einsum(self, other)
        if isinstance(other, GetIndicesOp):
            # Compose: self gathers from other's output
            if self.dim == other.dim:
                # Compose indices: self.indices indexes into other's output along dim,
                # which is other.added_shape. Map through other.indices.
                new_indices = other.indices.flatten()[self.indices.flatten()].reshape(self.added_shape)
                return GetIndicesOp(other.input_shape, self.dim, new_indices, self.added_shape)
        return NotImplemented

    def __str__(self):
        return f"<getindices dim={self.dim} added={list(self.added_shape)}>"


class SetIndicesOp(LinearOp):
    """Scatter values along *dim* using an index tensor.

    input_shape = output_shape[:dim] + added_shape + output_shape[dim+1:]

    Args:
        output_shape: Shape of the output tensor (zeros template).
        dim: Dimension along which to scatter.
        indices: Index tensor with shape ``added_shape``, values in ``[0, output_shape[dim])``.
        added_shape: Shape that replaces ``output_shape[dim]`` in the input.
    """

    def __init__(self, output_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size):
        self.dim = dim
        self.indices = indices
        self.added_shape = added_shape
        assert indices.shape == added_shape, \
            f"indices.shape={indices.shape} != added_shape={added_shape}"
        input_shape = torch.Size(
            list(output_shape[:dim]) + list(added_shape) + list(output_shape[dim + 1:])
        )
        super().__init__(input_shape, output_shape,
                         flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scatter(x, self.dim, self.output_shape[self.dim])

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return self._gather(grad, self.dim)

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scatter(x, self.dim, self.output_shape[self.dim])

    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        batch_ndim = grad.dim() - len(self.output_shape)
        return self._gather(grad, batch_ndim + self.dim)

    def _gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Gather along dim (backward of scatter)."""
        n_added = len(self.added_shape)
        idx = self.indices.reshape(-1)
        n_after = x.dim() - dim - 1
        for _ in range(n_after):
            idx = idx.unsqueeze(-1)
        for _ in range(dim):
            idx = idx.unsqueeze(0)
        idx = idx.expand(*x.shape[:dim], len(self.indices.flatten()), *x.shape[dim + 1:])
        result = torch.gather(x, dim, idx)
        if n_added > 1:
            result = result.unflatten(dim, self.added_shape)
        return result

    def _scatter(self, x: torch.Tensor, dim: int, target_size: int) -> torch.Tensor:
        """Scatter x along dim into zeros."""
        n_added = len(self.added_shape)
        if n_added > 1:
            x = x.flatten(dim, dim + n_added - 1)
        idx = self.indices.reshape(-1)
        n_after = x.dim() - dim - 1
        for _ in range(n_after):
            idx = idx.unsqueeze(-1)
        for _ in range(dim):
            idx = idx.unsqueeze(0)
        idx = idx.expand_as(x)
        result_shape = list(x.shape)
        result_shape[dim] = target_size
        result = torch.zeros(result_shape, dtype=x.dtype, device=x.device)
        result.scatter_add_(dim, idx, x)
        return result

    def __rmatmul__(self, other):
        """Fuse EinsumOp @ SetIndicesOp."""
        from boundlab.linearop._einsum import EinsumOp
        if isinstance(other, EinsumOp) and self.output_shape == other.input_shape:
            tensor_dim = other.input_dims[self.dim]
            if tensor_dim in other.mul_dims:
                return _swap_einsum_setindices(other, self)
            else:
                return _fuse_einsum_setindices(other, self)
        return super().__rmatmul__(other)

    def __str__(self):
        return f"<setindices dim={self.dim} added={list(self.added_shape)}>"


# ---------------------------------------------------------------------------
# Fusion with EinsumOp
# ---------------------------------------------------------------------------


def _index_tensor_along_dim(tensor, tensor_dim, indices):
    """Index a tensor along tensor_dim using 1D indices."""
    idx = indices.flatten()
    slices = [slice(None)] * tensor.dim()
    slices[tensor_dim] = idx
    result = tensor[tuple(slices)]
    if len(indices.shape) > 1:
        # Need to unflatten the tensor_dim
        shape = list(result.shape)
        shape[tensor_dim:tensor_dim + 1] = list(indices.shape)
        result = result.reshape(shape)
    return result


def _fuse_getindices_einsum(gi: GetIndicesOp, einsum):
    """Fuse GetIndicesOp @ EinsumOp when indexed dim is dot/batch."""
    from boundlab.linearop._einsum import EinsumOp

    tensor_dim = einsum.output_dims[gi.dim]
    new_tensor = _index_tensor_along_dim(einsum.tensor, tensor_dim, gi.indices)

    # Update output_dims: dim is replaced by added_shape dims
    n_added = len(gi.added_shape)
    n_replace = 1
    # The tensor now has added_shape dims instead of 1 dim at tensor_dim
    # But we indexed along tensor_dim, so tensor_dim now has size len(indices.flatten())
    # If added_shape is multi-dim, we need to reshape

    if n_added == 1:
        return EinsumOp(new_tensor, einsum.input_dims, einsum.output_dims,
                        name=merge_name(gi, "@", einsum))
    else:
        # Multi-dim added_shape: need to unflatten tensor_dim
        shape = list(new_tensor.shape)
        shape[tensor_dim:tensor_dim + 1] = list(gi.added_shape)
        new_tensor = new_tensor.reshape(shape)

        # Remap dims: dims after tensor_dim shift by (n_added - 1)
        shift = n_added - 1
        new_output_dims = []
        for i, d in enumerate(einsum.output_dims):
            if i < gi.dim:
                new_output_dims.append(d if d < tensor_dim else d + shift)
            elif i == gi.dim:
                for k in range(n_added):
                    new_output_dims.append(tensor_dim + k)
            else:
                new_output_dims.append(d + shift if d >= tensor_dim else d)
        new_input_dims = [d + shift if d > tensor_dim else d for d in einsum.input_dims]

        return EinsumOp(new_tensor, new_input_dims, new_output_dims,
                        name=merge_name(gi, "@", einsum))


def _swap_getindices_einsum(gi: GetIndicesOp, einsum):
    """Swap GetIndicesOp past EinsumOp for mul dim."""
    from boundlab.linearop._einsum import EinsumOp

    tensor_dim = einsum.output_dims[gi.dim]
    new_tensor = _index_tensor_along_dim(einsum.tensor, tensor_dim, gi.indices)

    # Find the input dim that shares tensor_dim
    input_d = einsum.input_dims.index(tensor_dim)

    if len(gi.added_shape) == 1:
        new_einsum = EinsumOp(new_tensor, einsum.input_dims, einsum.output_dims,
                              name=merge_name(gi, "@swap@", einsum))
    else:
        shape = list(new_tensor.shape)
        shape[tensor_dim:tensor_dim + 1] = list(gi.added_shape)
        new_tensor = new_tensor.reshape(shape)
        shift = len(gi.added_shape) - 1
        new_output_dims = []
        for i, d in enumerate(einsum.output_dims):
            if i < gi.dim:
                new_output_dims.append(d if d < tensor_dim else d + shift)
            elif i == gi.dim:
                for k in range(len(gi.added_shape)):
                    new_output_dims.append(tensor_dim + k)
            else:
                new_output_dims.append(d + shift if d >= tensor_dim else d)
        new_input_dims = list(einsum.input_dims)
        for j in range(len(new_input_dims)):
            if new_input_dims[j] > tensor_dim:
                new_input_dims[j] += shift
            elif new_input_dims[j] == tensor_dim:
                # This input dim now corresponds to the flattened added_shape
                # We'll handle this via a GetIndicesOp on input
                pass
        new_einsum = EinsumOp(new_tensor, new_input_dims, new_output_dims,
                              name=merge_name(gi, "@swap@", einsum))

    # Add GetIndicesOp on input side
    input_gi = GetIndicesOp(einsum.input_shape, input_d, gi.indices, gi.added_shape)
    return ComposedOp(new_einsum, input_gi)


def _fuse_einsum_setindices(einsum, si: SetIndicesOp):
    """Fuse EinsumOp @ SetIndicesOp when indexed dim is dot/batch."""
    from boundlab.linearop._einsum import EinsumOp

    tensor_dim = einsum.input_dims[si.dim]
    new_tensor = _index_tensor_along_dim(einsum.tensor, tensor_dim, si.indices)

    if len(si.added_shape) == 1:
        return EinsumOp(new_tensor, einsum.input_dims, einsum.output_dims,
                        name=merge_name(einsum, "@", si))
    else:
        shape = list(new_tensor.shape)
        shape[tensor_dim:tensor_dim + 1] = list(si.added_shape)
        new_tensor = new_tensor.reshape(shape)
        shift = len(si.added_shape) - 1
        new_input_dims = []
        for i, d in enumerate(einsum.input_dims):
            if i < si.dim:
                new_input_dims.append(d if d < tensor_dim else d + shift)
            elif i == si.dim:
                for k in range(len(si.added_shape)):
                    new_input_dims.append(tensor_dim + k)
            else:
                new_input_dims.append(d + shift if d >= tensor_dim else d)
        new_output_dims = [d + shift if d > tensor_dim else d for d in einsum.output_dims]
        return EinsumOp(new_tensor, new_input_dims, new_output_dims,
                        name=merge_name(einsum, "@", si))


def _swap_einsum_setindices(einsum, si: SetIndicesOp):
    """Swap SetIndicesOp past EinsumOp for mul dim."""
    from boundlab.linearop._einsum import EinsumOp

    tensor_dim = einsum.input_dims[si.dim]
    new_tensor = _index_tensor_along_dim(einsum.tensor, tensor_dim, si.indices)
    output_d = einsum.output_dims.index(tensor_dim)

    if len(si.added_shape) == 1:
        new_einsum = EinsumOp(new_tensor, einsum.input_dims, einsum.output_dims,
                              name=merge_name(einsum, "@swap@", si))
    else:
        shape = list(new_tensor.shape)
        shape[tensor_dim:tensor_dim + 1] = list(si.added_shape)
        new_tensor = new_tensor.reshape(shape)
        shift = len(si.added_shape) - 1
        new_input_dims = []
        for i, d in enumerate(einsum.input_dims):
            if i < si.dim:
                new_input_dims.append(d if d < tensor_dim else d + shift)
            elif i == si.dim:
                for k in range(len(si.added_shape)):
                    new_input_dims.append(tensor_dim + k)
            else:
                new_input_dims.append(d + shift if d >= tensor_dim else d)
        new_output_dims = list(einsum.output_dims)
        for j in range(len(new_output_dims)):
            if new_output_dims[j] > tensor_dim:
                new_output_dims[j] += shift
        new_einsum = EinsumOp(new_tensor, new_input_dims, new_output_dims,
                              name=merge_name(einsum, "@swap@", si))

    output_si = SetIndicesOp(einsum.output_shape, output_d, si.indices, si.added_shape)
    return ComposedOp(output_si, new_einsum)
