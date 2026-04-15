"""Miscellaneous shape LinearOp implementations.

Contains ops not covered by the dedicated reshape/permute/expand modules:
RepeatOp, TileOp, FlipOp, RollOp, DiagOp.

Re-exports reshape/permute/expand ops for backward compatibility.
"""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags

# Re-export for backward compatibility
from boundlab.linearop._reshape import (
    ReshapeOp, FlattenOp, UnflattenOp, SqueezeOp, UnsqueezeOp, _meta_output_shape,
)
from boundlab.linearop._permute import PermuteOp, TransposeOp
from boundlab.linearop._expand import ExpandOp


# ---------------------------------------------------------------------------
# Repeat / tile
# ---------------------------------------------------------------------------

class RepeatOp(LinearOp):
    """Tile-repeat the tensor (adjoint folds and sums repeated blocks)."""

    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        self.sizes = sizes
        n_pad = len(sizes) - len(input_shape)
        self._padded_input_shape = torch.Size([1] * n_pad + list(input_shape))
        output_shape = torch.Size(
            s * r for s, r in zip(self._padded_input_shape, sizes))
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x):
        return x.repeat(*self.sizes)

    def backward(self, grad):
        new_shape = []
        for r, s in zip(self.sizes, self._padded_input_shape):
            new_shape.extend([r, s])
        grad = grad.reshape(new_shape)
        sum_dims = list(range(0, len(new_shape), 2))
        grad = grad.sum(dim=sum_dims)
        return grad.reshape(self.input_shape)

    def __str__(self):
        return f"<repeat {list(self.sizes)}>"


class TileOp(RepeatOp):
    """Alias for repeat with dimension-padding handled like ``torch.tile``."""

    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        n_pad = len(input_shape) - len(sizes)
        if n_pad > 0:
            sizes = (1,) * n_pad + tuple(sizes)
        super().__init__(input_shape, sizes)

    def __str__(self):
        return f"<tile {list(self.sizes)}>"


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
        return f"<flip {self.dims}>"


class RollOp(LinearOp):
    """Circular-shift elements (adjoint is the inverse shift)."""

    def __init__(self, input_shape: torch.Size, shifts, dims):
        self.shifts = shifts
        self.dims = dims
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
        return f"<roll {self.shifts} {self.dims}>"


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
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.diag(self.diagonal)

    def backward(self, grad):
        if self._input_ndim == 1:
            return grad.diag(self.diagonal)
        else:
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
        return f"<diag {self.diagonal}>"
