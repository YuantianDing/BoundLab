"""Shape and indexing LinearOp implementations for bound propagation.

Each class implements explicit forward (the shape operation) and backward
(its adjoint/transpose) so that no automatic VJP is needed.
"""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags, ScalarOp


def _meta_output_shape(fn, input_shape: torch.Size) -> torch.Size:
    """Compute output shape by tracing *fn* on a meta-device tensor."""
    return fn(torch.empty(input_shape, device="meta")).shape


# ---------------------------------------------------------------------------
# Reshape / view operations
# ---------------------------------------------------------------------------

class ReshapeOp(LinearOp):
    """Reshape (view) the input tensor to *target_shape*."""

    def __init__(self, input_shape: torch.Size, target_shape: tuple[int, ...]):
        self.target_shape = target_shape
        output_shape = _meta_output_shape(lambda x: x.reshape(*target_shape), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.reshape(*self.target_shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

    def vforward(self, x):
        extra = x.shape[len(self.input_shape):]
        return x.reshape(*self.output_shape, *extra)

    def vbackward(self, grad):
        extra = grad.shape[:-len(self.output_shape)]
        return grad.reshape(*extra, *self.input_shape)

    def __str__(self):
        return f"reshape({list(self.target_shape)})"


class FlattenOp(LinearOp):
    """Flatten dimensions [start_dim .. end_dim] into a single dimension."""

    def __init__(self, input_shape: torch.Size, start_dim: int = 0, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim if end_dim >= 0 else len(input_shape) + end_dim
        self.original_sizes = input_shape[self.start_dim:self.end_dim + 1]
        output_shape = _meta_output_shape(
            lambda x: x.flatten(start_dim, end_dim), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)

    def backward(self, grad):
        return grad.unflatten(self.start_dim, self.original_sizes)

    def __str__(self):
        return f"flatten({self.start_dim}, {self.end_dim})"


class UnflattenOp(LinearOp):
    """Unflatten dimension *dim* into *sizes*."""

    def __init__(self, input_shape: torch.Size, dim: int, sizes: tuple[int, ...]):
        self.dim = dim
        self.sizes = sizes
        self.end_dim = dim + len(sizes) - 1
        output_shape = _meta_output_shape(
            lambda x: x.unflatten(dim, sizes), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.unflatten(self.dim, self.sizes)

    def backward(self, grad):
        return grad.flatten(self.dim, self.end_dim)

    def __str__(self):
        return f"unflatten({self.dim}, {list(self.sizes)})"


# ---------------------------------------------------------------------------
# Permutation / transposition
# ---------------------------------------------------------------------------

class PermuteOp(LinearOp):
    """Permute dimensions of the input tensor."""

    def __init__(self, input_shape: torch.Size, dims: tuple[int, ...]):
        self.dims = list(dims)
        self.inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            self.inv_dims[d] = i
        output_shape = torch.Size(input_shape[d] for d in dims)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.permute(*self.dims)

    def backward(self, grad):
        return grad.permute(*self.inv_dims)

    def vforward(self, x):
        n = len(self.dims)
        batch_ndim = x.dim() - n
        perm = self.dims + [n + i for i in range(batch_ndim)]
        return x.permute(*perm)

    def vbackward(self, grad):
        n = len(self.inv_dims)
        batch_ndim = grad.dim() - n
        perm = list(range(batch_ndim)) + [batch_ndim + d for d in self.inv_dims]
        return grad.permute(*perm)

    def __str__(self):
        return f"permute({self.dims})"


class TransposeOp(PermuteOp):
    """Swap two dimensions of the input tensor — special case of PermuteOp."""

    def __init__(self, input_shape: torch.Size, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1
        dims = list(range(len(input_shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        super().__init__(input_shape, tuple(dims))

    def __str__(self):
        return f"transpose({self.dim0}, {self.dim1})"


# ---------------------------------------------------------------------------
# Squeeze / unsqueeze
# ---------------------------------------------------------------------------

class SqueezeOp(LinearOp):
    """Remove size-1 dimension(s)."""

    def __init__(self, input_shape: torch.Size, dim=None):
        self.dim = dim
        if dim is not None:
            self._is_noop = (input_shape[dim] != 1)
            if self._is_noop:
                output_shape = input_shape
            else:
                output_shape = torch.Size(
                    s for i, s in enumerate(input_shape) if i != dim)
        else:
            self._is_noop = all(s != 1 for s in input_shape)
            self._squeezed_dims = [i for i, s in enumerate(input_shape) if s == 1]
            output_shape = torch.Size(s for s in input_shape if s != 1)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.squeeze(self.dim) if self.dim is not None else x.squeeze()

    def backward(self, grad):
        if self._is_noop:
            return grad
        if self.dim is not None:
            return grad.unsqueeze(self.dim)
        for d in self._squeezed_dims:
            grad = grad.unsqueeze(d)
        return grad

    def __str__(self):
        return f"squeeze({self.dim})"


class UnsqueezeOp(LinearOp):
    """Insert a size-1 dimension at *dim*."""

    def __init__(self, input_shape: torch.Size, dim: int):
        self.dim = dim
        output_shape = _meta_output_shape(lambda x: x.unsqueeze(dim), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.unsqueeze(self.dim)

    def backward(self, grad):
        return grad.squeeze(self.dim)

    def vforward(self, x):
        return x.unsqueeze(self.dim)

    def vbackward(self, grad):
        batch_ndim = grad.dim() - len(self.output_shape)
        return grad.squeeze(batch_ndim + self.dim)

    def __str__(self):
        return f"unsqueeze({self.dim})"


# ---------------------------------------------------------------------------
# Expand / repeat / tile
# ---------------------------------------------------------------------------

class ExpandOp(LinearOp):
    """Broadcast-expand dimensions (adjoint sums over expanded dims)."""

    def __new__(cls, input_shape: torch.Size, sizes: tuple[int, ...]):
        if input_shape == torch.Size(sizes):
            return ScalarOp(1.0, input_shape)
        return super().__new__(cls)

    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        self.sizes = sizes
        output_shape = _meta_output_shape(lambda x: x.expand(*sizes), input_shape)
        # Dims that need to be summed in backward
        n_new = len(output_shape) - len(input_shape)
        self._sum_dims: list[int] = list(range(n_new))
        for i in range(len(input_shape)):
            if input_shape[i] == 1 and output_shape[n_new + i] > 1:
                self._sum_dims.append(n_new + i)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.expand(*self.sizes)

    def backward(self, grad):
        if self._sum_dims:
            grad = grad.sum(dim=self._sum_dims)
        return grad.reshape(self.input_shape)

    def __str__(self):
        return f"expand({list(self.sizes)})"


class RepeatOp(LinearOp):
    """Tile-repeat the tensor (adjoint folds and sums repeated blocks)."""

    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        self.sizes = sizes
        n_pad = len(sizes) - len(input_shape)
        self._padded_input_shape = torch.Size([1] * n_pad + list(input_shape))
        output_shape = torch.Size(
            s * r for s, r in zip(self._padded_input_shape, sizes))
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.repeat(*self.sizes)

    def backward(self, grad):
        # Interleave (repeat_factor, original_size) pairs, then sum repeat dims
        new_shape = []
        for r, s in zip(self.sizes, self._padded_input_shape):
            new_shape.extend([r, s])
        grad = grad.reshape(new_shape)
        sum_dims = list(range(0, len(new_shape), 2))
        grad = grad.sum(dim=sum_dims)
        return grad.reshape(self.input_shape)

    def __str__(self):
        return f"repeat({list(self.sizes)})"


class TileOp(RepeatOp):
    """Alias for repeat with dimension-padding handled like ``torch.tile``."""

    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        # tile pads *sizes* with leading 1s when tensor has more dims
        n_pad = len(input_shape) - len(sizes)
        if n_pad > 0:
            sizes = (1,) * n_pad + tuple(sizes)
        super().__init__(input_shape, sizes)

    def __str__(self):
        return f"tile({list(self.sizes)})"


# ---------------------------------------------------------------------------
# Element-reordering (self-adjoint or simple inverse)
# ---------------------------------------------------------------------------

class FlipOp(LinearOp):
    """Reverse elements along *dims* (self-adjoint)."""

    def __init__(self, input_shape: torch.Size, dims):
        self.dims = dims
        super().__init__(input_shape, input_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.flip(self.dims)

    def backward(self, grad):
        return grad.flip(self.dims)

    def __str__(self):
        return f"flip({self.dims})"


class RollOp(LinearOp):
    """Circular-shift elements (adjoint is the inverse shift)."""

    def __init__(self, input_shape: torch.Size, shifts, dims):
        self.shifts = shifts
        self.dims = dims
        # Inverse shifts for backward
        if isinstance(shifts, int):
            self._inv_shifts = -shifts
        else:
            self._inv_shifts = [-s for s in shifts]
        super().__init__(input_shape, input_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.roll(self.shifts, self.dims)

    def backward(self, grad):
        return grad.roll(self._inv_shifts, self.dims)

    def __str__(self):
        return f"roll({self.shifts}, {self.dims})"


# ---------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------

class DiagOp(LinearOp):
    """Extract or create a diagonal (1D↔2D)."""

    def __init__(self, input_shape: torch.Size, diagonal: int = 0):
        self.diagonal = diagonal
        self._input_ndim = len(input_shape)
        output_shape = _meta_output_shape(
            lambda x: x.diag(diagonal), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.diag(self.diagonal)

    def backward(self, grad):
        if self._input_ndim == 1:
            # Forward was 1D→2D (create diagonal matrix); adjoint extracts diagonal
            return grad.diag(self.diagonal)
        else:
            # Forward was 2D→1D (extract diagonal); adjoint embeds into zeros
            result = torch.zeros(
                self.input_shape, dtype=grad.dtype, device=grad.device)
            n = len(grad)
            idx = torch.arange(n, device=grad.device)
            if self.diagonal >= 0:
                result[idx, idx + self.diagonal] = grad
            else:
                result[idx - self.diagonal, idx] = grad
            return result

    def __str__(self):
        return f"diag({self.diagonal})"


# ---------------------------------------------------------------------------
# Reduction operations
# ---------------------------------------------------------------------------

class ReduceMeanOp(LinearOp):
    """Mean reduction along one or more axes.

    Forward:  ``y = x.mean(dims, keepdim=keepdim)``
    Backward: uniform expansion ``grad.expand(input_shape) / n``
    """

    def __init__(self, input_shape: torch.Size, dims: tuple[int, ...], keepdim: bool = False):
        self.dims = tuple(d % len(input_shape) for d in dims)
        self.keepdim = keepdim
        self.n = 1
        for d in self.dims:
            self.n *= input_shape[d]
        output_shape = _meta_output_shape(
            lambda x: x.mean(dim=self.dims, keepdim=keepdim), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.NONE)

    def forward(self, x):
        return x.mean(dim=self.dims, keepdim=self.keepdim)

    def backward(self, grad):
        g = grad
        if not self.keepdim:
            for d in sorted(self.dims):
                g = g.unsqueeze(d)
        return g.expand(self.input_shape) / self.n

    def vforward(self, x):
        extra = x.shape[len(self.input_shape):]
        return x.mean(dim=self.dims, keepdim=self.keepdim).reshape(*self.output_shape, *extra)

    def vbackward(self, grad):
        extra = grad.shape[:-len(self.output_shape)] if len(self.output_shape) > 0 else grad.shape
        g = grad.reshape(*extra, *self.output_shape)
        if not self.keepdim:
            for d in sorted(self.dims):
                g = g.unsqueeze(len(extra) + d)
        return g.expand(*extra, *self.input_shape) / self.n

    def __str__(self):
        return f"reduce_mean(dims={self.dims}, keepdim={self.keepdim})"


