"""Reshape LinearOp implementations.

ReshapeOp is the base class; FlattenOp, UnflattenOp, SqueezeOp, and
UnsqueezeOp are thin subclasses that delegate to ReshapeOp logic.
"""

from functools import reduce

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags, ScalarOp
from boundlab.utils import merge_name


def _meta_output_shape(fn, input_shape: torch.Size) -> torch.Size:
    """Compute output shape by tracing *fn* on a meta-device tensor."""
    return fn(torch.empty(input_shape, device="meta")).shape


class ReshapeOp(LinearOp):
    """Reshape (view) the input tensor to *output_shape*."""

    def __init__(self, input_shape: torch.Size, output_shape: tuple[int, ...]):
        if not isinstance(output_shape, torch.Size):
            output_shape = _meta_output_shape(lambda x: x.reshape(*output_shape), input_shape)
        self.target_shape = tuple(output_shape)
        self.reshape_groups = []
        self.dims_map = {}
        self.dims_map_inv = {}

        # Edge case: either side is a scalar (0-dim). Allowed only when
        # both sides have numel == 1.
        if len(input_shape) == 0 or len(output_shape) == 0:
            input_numel = reduce(lambda x, y: x * y, input_shape, 1)
            output_numel = reduce(lambda x, y: x * y, output_shape, 1)
            assert input_numel == output_numel, \
                f"ReshapeOp: cannot align {input_shape} -> {output_shape} (numel {input_numel} != {output_numel})"
            if len(input_shape) > 0 or len(output_shape) > 0:
                self.reshape_groups.append((0, len(input_shape) - 1, 0, len(output_shape) - 1))
            super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)
            return

        i, j = 0, 0
        input_win = 0
        output_win = 0

        while True:
            input_numel = reduce(lambda x, y: x * y, input_shape[input_win:i+1], 1)
            output_numel = reduce(lambda x, y: x * y, output_shape[output_win:j+1], 1)
            if input_numel == output_numel:
                if i > input_win or j > output_win:
                    self.reshape_groups.append((input_win, i, output_win, j))
                else:
                    assert i == input_win and j == output_win
                    self.dims_map[i] = j
                    self.dims_map_inv[j] = i
                i += 1
                j += 1
                input_win = i
                output_win = j

            if input_numel < output_numel:
                i += 1

            if input_numel > output_numel:
                j += 1

            if i >= len(input_shape) or j >= len(output_shape):
                while i < len(input_shape) and input_shape[i] == 1:
                    i += 1
                while j < len(output_shape) and output_shape[j] == 1:
                    j += 1
                if input_win < i or output_win < j:
                    self.reshape_groups.append((input_win, max(i - 1, input_win), output_win, max(j - 1, output_win)))
                assert i == len(input_shape) and j == len(output_shape), \
                    f"ReshapeOp: cannot align {input_shape} -> {output_shape} (i={i}, j={j})"
                break

        # TODO
    def __str__(self):
        return f"<reshape {list(self.input_shape)} -> {list(self.target_shape)}>"


class FlattenOp(ReshapeOp):
    """Flatten dimensions [start_dim .. end_dim] into a single dimension."""

    def __init__(self, input_shape: torch.Size, start_dim: int = 0, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim if end_dim >= 0 else len(input_shape) + end_dim
        self.original_sizes = input_shape[self.start_dim:self.end_dim + 1]
        target_shape = _meta_output_shape(
            lambda x: x.flatten(start_dim, end_dim), input_shape)
        super().__init__(input_shape, target_shape)

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)

    def backward(self, grad):
        return grad.unflatten(self.start_dim, self.original_sizes)

    def __str__(self):
        return f"<flatten {self.start_dim} {self.end_dim}>"


class UnflattenOp(ReshapeOp):
    """Unflatten dimension *dim* into *sizes*."""

    def __init__(self, input_shape: torch.Size, dim: int, sizes: tuple[int, ...]):
        self.dim = dim
        self.sizes = sizes
        self.end_dim = dim + len(sizes) - 1
        target_shape = _meta_output_shape(
            lambda x: x.unflatten(dim, sizes), input_shape)
        super().__init__(input_shape, target_shape)

    def __str__(self):
        return f"<unflatten {self.dim} {list(self.sizes)}>"


class SqueezeOp(ReshapeOp):
    """Remove size-1 dimension(s)."""

    def __init__(self, input_shape: torch.Size, dim=None):
        self.dim = dim
        if dim is not None:
            self._is_noop = (input_shape[dim] != 1)
            if self._is_noop:
                target_shape = input_shape
            else:
                target_shape = torch.Size(
                    s for i, s in enumerate(input_shape) if i != dim)
        else:
            self._is_noop = all(s != 1 for s in input_shape)
            self._squeezed_dims = [i for i, s in enumerate(input_shape) if s == 1]
            target_shape = torch.Size(s for s in input_shape if s != 1)
        super().__init__(input_shape, target_shape)

    def __str__(self):
        return f"<squeeze {self.dim}>"


class UnsqueezeOp(ReshapeOp):
    """Insert a size-1 dimension at *dim*."""

    def __init__(self, input_shape: torch.Size, dim: int):
        if dim < 0:
            dim += len(input_shape) + 1
        assert 0 <= dim <= len(input_shape), f"Invalid unsqueeze dim {dim} for input shape {input_shape}"
        self.dim = dim
        target_shape = _meta_output_shape(lambda x: x.unsqueeze(dim), input_shape)
        super().__init__(input_shape, target_shape)

    def __str__(self):
        return f"<unsqueeze {list(self.input_shape)} {self.dim}>"
