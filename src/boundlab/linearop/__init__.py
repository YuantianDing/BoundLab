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

from ._base import LinearOp, ComposedOp, SumOp, ScalarOp, ZeroOp
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
    narrow_indices,
    select_indices,
    pad_indices,
    pad_output_shape,
    _format_indices,
    make_get_slices,
    make_set_slices,
    get_int_dims,
)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases (convenience wrappers)
# ---------------------------------------------------------------------------

class NarrowOp(GetSliceOp):
    """Select a contiguous slice along *dim*. (Alias for GetSliceOp)

    Examples
    --------
    >>> import torch
    >>> from boundlab.linearop import NarrowOp
    >>> op = NarrowOp(torch.Size([5]), dim=0, start=1, length=3)
    >>> op.forward(torch.tensor([0., 1., 2., 3., 4.]))
    tensor([1., 2., 3.])
    """

    def __init__(self, input_shape, dim: int, start: int, length: int):
        ndim = len(input_shape)
        slices = [[slice(0, input_shape[d])] for d in range(ndim)]
        slices[dim] = [slice(start, start + length)]
        super().__init__(input_shape, slices)
        self.dim = dim
        self.start = start
        self.length = length

    def __str__(self):
        return f"<narrow {self.dim} {self.start} {self.length}>"


class SelectOp(LinearOp):
    """Select a single index along *dim*, removing that dimension.
    
    Implemented as GetSliceOp (length-1 slice) composed with SqueezeOp.

    Examples
    --------
    >>> import torch
    >>> from boundlab.linearop import SelectOp
    >>> op = SelectOp(torch.Size([2, 3]), dim=0, index=1)
    >>> op.forward(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    tensor([4., 5., 6.])
    """

    def __init__(self, input_shape, dim: int, index: int):
        from ._base import LinearOpFlags
        self.dim = dim
        self.index = index
        if index < 0:
            index += input_shape[dim]
        ndim = len(input_shape)
        slices = [[slice(0, input_shape[d])] for d in range(ndim)]
        slices[dim] = [slice(index, index + 1)]
        self._slice_op = GetSliceOp(input_shape, slices)
        self._squeeze_op = SqueezeOp(self._slice_op.output_shape, dim=dim)
        output_shape = self._squeeze_op.output_shape
        super().__init__(input_shape, output_shape,
                         flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return self._squeeze_op.forward(self._slice_op.forward(x))

    def backward(self, grad):
        return self._slice_op.backward(self._squeeze_op.backward(grad))

    def vforward(self, x):
        return self._squeeze_op.vforward(self._slice_op.vforward(x))

    def vbackward(self, grad):
        return self._slice_op.vbackward(self._squeeze_op.vbackward(grad))

    def __matmul__(self, other):
        composed = ComposedOp(self._squeeze_op, self._slice_op)
        return composed @ other

    def __rmatmul__(self, other):
        composed = ComposedOp(self._squeeze_op, self._slice_op)
        return other @ composed

    def __str__(self):
        return f"<select {self.dim}, {self.index}>"


class GetItemOp(LinearOp):
    """Indexing / slicing via ``x[indices]``.
    
    Handles both slice and int indices. Int indices remove dims (via squeeze).
    """

    def __init__(self, input_shape, indices):
        from ._base import LinearOpFlags
        self.indices = indices
        # Convert to structured slices + track int dims for squeezing
        self._int_dims = get_int_dims(indices)
        slices = make_get_slices(input_shape, indices)
        self._slice_op = GetSliceOp(input_shape, slices)
        
        # Build squeeze chain for int dims
        intermediate_shape = self._slice_op.output_shape
        self._squeeze_ops = []
        for i, dim in enumerate(sorted(self._int_dims)):
            adjusted_dim = dim - i  # earlier squeezes shift later dims
            sq = SqueezeOp(intermediate_shape, dim=adjusted_dim)
            self._squeeze_ops.append(sq)
            intermediate_shape = sq.output_shape
        
        output_shape = intermediate_shape
        super().__init__(input_shape, output_shape,
                         flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        x = self._slice_op.forward(x)
        for sq in self._squeeze_ops:
            x = sq.forward(x)
        return x

    def backward(self, grad):
        for sq in reversed(self._squeeze_ops):
            grad = sq.backward(grad)
        return self._slice_op.backward(grad)

    def vforward(self, x):
        x = self._slice_op.vforward(x)
        for sq in self._squeeze_ops:
            x = sq.vforward(x)
        return x

    def vbackward(self, grad):
        for sq in reversed(self._squeeze_ops):
            grad = sq.vbackward(grad)
        return self._slice_op.vbackward(grad)

    def __matmul__(self, other):
        ops = [self._slice_op] + self._squeeze_ops
        composed = ComposedOp(*reversed(ops))
        return composed @ other

    def __rmatmul__(self, other):
        ops = [self._slice_op] + self._squeeze_ops
        composed = ComposedOp(*reversed(ops))
        return other @ composed

    def __str__(self):
        return f"<getitem {_format_indices(self.indices)}>"


class PadOp(SetSliceOp):
    """Zero-pad an input tensor. (Alias for SetSliceOp)

    Examples
    --------
    >>> import torch
    >>> from boundlab.linearop import PadOp
    >>> op = PadOp(torch.Size([3]), [1, 2])
    >>> op.forward(torch.tensor([1., 2., 3.]))
    tensor([0., 1., 2., 3., 0., 0.])
    """

    def __init__(self, input_shape, pad_spec: list[int]):
        self._pad_spec = list(pad_spec)
        output_shape = pad_output_shape(input_shape, pad_spec)
        ndim = len(output_shape)
        slices = []
        for d in range(ndim):
            d_rev = ndim - 1 - d
            if 2 * d_rev + 1 < len(pad_spec):
                pad_before = pad_spec[2 * d_rev]
                slices.append([slice(pad_before, pad_before + input_shape[d])])
            else:
                slices.append([slice(0, output_shape[d])])
        super().__init__(output_shape, slices)

    def __str__(self):
        return f"<pad {self._pad_spec}>"


__all__ = [
    # Base classes
    "LinearOp",
    "ComposedOp",
    "SumOp",
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
