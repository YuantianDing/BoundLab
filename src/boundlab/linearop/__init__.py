r"""Linear Operator Library for Expression Backpropagation.

This module defines linear operators used by BoundLab expressions during
symbolic transformation and backward bound propagation.

Key operators:

- :class:`ComposedOp`: Functional composition of linear maps
  (:math:`A \circ B`), used to chain transformations efficiently.
- :class:`SumOp`: Pointwise sum of linear maps, used when multiple affine
  contributions target the same expression.
- :class:`EinsumOp`: General tensor-linear map based on Einstein summation;
  this is the most flexible primitive for dense affine transformations.

The module also exposes shape/indexing operators (reshape, permute, gather,
scatter, slicing, padding) that are all represented as :class:`LinearOp`
instances and can therefore be composed, summed, and propagated uniformly.
"""

from ._base import LinearOp, ComposedOp, SumOp, ScalarOp, ZeroOp
from ._einsum import EinsumOp
from ._shape import (
    ReshapeOp,
    FlattenOp,
    UnflattenOp,
    PermuteOp,
    TransposeOp,
    SqueezeOp,
    UnsqueezeOp,
    ExpandOp,
    RepeatOp,
    TileOp,
    FlipOp,
    RollOp,
    DiagOp,
)
from ._indices import (
    # Core indexing ops
    GatherOp,
    ScatterOp,
    GetIndicesOp,
    SetIndicesOp,
    GetSliceOp,
    SetSliceOp,
    # Convenience functions
    narrow_indices,
    select_indices,
    pad_indices,
    pad_output_shape,
)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases (convenience wrappers)
# ---------------------------------------------------------------------------

class NarrowOp(GetSliceOp):
    """Select a contiguous slice along *dim*. (Alias for GetSliceOp)"""

    def __init__(self, input_shape, dim: int, start: int, length: int):
        indices = narrow_indices(len(input_shape), dim, start, length)
        super().__init__(input_shape, indices)
        self.dim = dim
        self.start = start
        self.length = length

    def __str__(self):
        return f"narrow({self.dim}, {self.start}, {self.length})"


class SelectOp(GetSliceOp):
    """Select a single index along *dim*, removing that dimension. (Alias for GetSliceOp)"""

    def __init__(self, input_shape, dim: int, index: int):
        indices = select_indices(len(input_shape), dim, index)
        super().__init__(input_shape, indices)
        self.dim = dim
        self.index = index

    def __str__(self):
        return f"select({self.dim}, {self.index})"


class GetItemOp(GetSliceOp):
    """Indexing / slicing via ``x[indices]``. (Alias for GetSliceOp)"""

    def __str__(self):
        from ._indices import _format_indices
        return f"getitem({_format_indices(self.indices)})"



class PadOp(SetSliceOp):
    """Zero-pad an input tensor. (Alias for SetSliceOp)"""

    def __init__(self, input_shape, pad_spec: list[int]):
        self._pad_spec = list(pad_spec)
        output_shape = pad_output_shape(input_shape, pad_spec)
        indices = pad_indices(input_shape, pad_spec)
        super().__init__(indices, input_shape, output_shape)

    def __str__(self):
        return f"pad({self._pad_spec})"


__all__ = [
    # Base classes
    "LinearOp",
    "ComposedOp",
    "SumOp",
    "ScalarOp",
    "ZeroOp",
    # Einsum
    "EinsumOp",
    # Shape ops
    "ReshapeOp",
    "FlattenOp",
    "UnflattenOp",
    "PermuteOp",
    "TransposeOp",
    "SqueezeOp",
    "UnsqueezeOp",
    "ExpandOp",
    "RepeatOp",
    "TileOp",
    "FlipOp",
    "RollOp",
    "DiagOp",
    # Indexing ops (general)
    "GatherOp",
    "ScatterOp",
    "GetIndicesOp",
    "SetIndicesOp",
    "GetSliceOp",
    "SetSliceOp",
    # Convenience aliases
    "NarrowOp",
    "SelectOp",
    "GetItemOp",
    "PadOp",
    # Utility functions
    "narrow_indices",
    "select_indices",
    "pad_indices",
    "pad_output_shape",
]
