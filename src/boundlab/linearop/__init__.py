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

Examples
--------
Apply a shape operator to a concrete tensor:

>>> import torch
>>> from boundlab.linearop import ReshapeOp
>>> op = ReshapeOp(torch.Size([2, 3]), (3, 2))
>>> y = op.forward(torch.arange(6.0).reshape(2, 3))
>>> y.shape
torch.Size([3, 2])
"""

import torch

from ._base import LinearOp
from ._compat import ScalarOp, ZeroOp
from ._einsum import EinsumOp

# Reshape ops
from ._reshape import (
    ReshapeOp,
    FlattenOp,
    UnflattenOp,
    SqueezeOp,
    UnsqueezeOp,
)

# Permute ops
from ._permute import PermuteOp, TransposeOp

# Expand
from ._expand import ExpandOp

# Remaining shape ops
from ._shape import (
    RepeatOp,
    TileOp,
    FlipOp,
    RollOp,
    DiagOp,
)

# Slicing ops (new structured API)
from ._slicing import GetSliceOp, SetSliceOp

# Indexing ops (new dim-based API)
from ._indexing import GetIndicesOp, SetIndicesOp

# Gather/Scatter and convenience functions
from ._indices import (
    GatherOp,
    ScatterOp,
    NarrowOp,
    SelectOp,
    GetItemOp,
    PadOp,
    narrow_indices,
    select_indices,
    pad_indices,
    pad_output_shape,
    make_get_slices,
    make_set_slices,
    get_int_dims,
)

__all__ = [
    # Base classes
    "LinearOp",
    "ScalarOp",
    "ZeroOp",
    # Einsum
    "EinsumOp",
    # Reshape ops
    "ReshapeOp",
    "FlattenOp",
    "UnflattenOp",
    "SqueezeOp",
    "UnsqueezeOp",
    # Permute ops
    "PermuteOp",
    "TransposeOp",
    # Expand
    "ExpandOp",
    # Other shape ops
    "RepeatOp",
    "TileOp",
    "FlipOp",
    "RollOp",
    "DiagOp",
    # Slicing ops
    "GetSliceOp",
    "SetSliceOp",
    # Indexing ops
    "GetIndicesOp",
    "SetIndicesOp",
    # Gather/Scatter
    "GatherOp",
    "ScatterOp",
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
    "make_get_slices",
    "make_set_slices",
]
