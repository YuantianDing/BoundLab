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
        # TODO:

    def __str__(self):
        return f"<gather dim={self.dim} index.shape={list(self.index.shape)}>"


class ScatterOp(LinearOp):
    """A LinearOp that implements ``torch.scatter`` along a specified dimension."""

    def __init__(self, input_shape: torch.Size, dim: int, index: torch.Tensor, output_shape: torch.Size):
        self.dim = dim
        self.index = index
        assert index.shape == input_shape, f"Index shape {index.shape} must match input shape {input_shape}"
        # TODO

    def __str__(self):
        return f"<scatter dim={self.dim} index.shape={list(self.index.shape)}>"

