r"""Concatenation and Stacking Operations

This module provides expressions for concatenating and stacking
child expressions along specified dimensions.
"""

from typing import Literal

import torch

from boundlab.expr._core import Expr, ExprFlags
from boundlab.expr._affine import AffineSum


def Cat(*children: Expr, dim: int = 0) -> AffineSum:
    """Concatenate child expressions along a dimension.

    Implemented as an :class:`AffineSum` whose per-child operator is a
    :class:`~boundlab.linearop.SetSliceOp` that embeds the child into the
    correct slice of the full output shape.
    """
    from boundlab.linearop import SetSliceOp

    assert all(isinstance(c, Expr) for c in children), "All children of Cat must be Expr instances."
    assert len(children) >= 1, "Cat requires at least one child."
    if dim < 0:
        dim += len(children[0].shape)
    assert all(
        c.shape[:dim] == children[0].shape[:dim] and c.shape[dim + 1:] == children[0].shape[dim + 1:]
        for c in children
    ), "All children must have matching shapes except along the concatenation dimension."

    cat_size = sum(c.shape[dim] for c in children)
    out_shape = list(children[0].shape)
    out_shape[dim] = cat_size
    out_shape = torch.Size(out_shape)

    pairs = []
    offset = 0
    for c in children:
        size = c.shape[dim]
        slices = [[slice(0, s)] for s in out_shape]
        slices[dim] = [slice(offset, offset + size)]
        pairs.append((SetSliceOp(out_shape, slices), c))
        offset += size
    return AffineSum(*pairs)


class Stack(Expr):
    """Expression for stacking child expressions along a new dimension.

    All children must have identical shapes. The backward pass produces
    an embed LinearOp per child that places the child at its index along
    the stacking dimension, with zeros elsewhere.
    """

    def __init__(self, *children: Expr, dim: int = 0):
        assert all(isinstance(c, Expr) for c in children), "All children of Stack must be Expr instances."
        super().__init__(ExprFlags.IS_AFFINE)
        assert len(children) >= 1, "Stack requires at least one child."
        assert all(
            c.shape == children[0].shape for c in children
        ), "All children must have the same shape for Stack."
        self._children = tuple(children)
        self.dim = dim

        s = list(children[0].shape)
        s.insert(dim, len(children))
        self._shape = torch.Size(s)

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def children(self) -> tuple[Expr, ...]:
        return self._children

    def with_children(self, *new_children: Expr) -> "Stack":
        return Stack(*new_children, dim=self.dim)

    def backward(self, weights, direction: Literal[">=", "<=", "=="] = "=="):  # noqa: ARG002
        """Propagate weights to each child via unsqueeze+cat embed ops.

        Args:
            weights: A :class:`~boundlab.linearop.EinsumOp` accumulated weight.
            direction: Unused (Stack is always linear).

        Returns:
            ``(0, [child_weight_0, child_weight_1, ...])``
        """
        from boundlab.linearop import PadOp, UnsqueezeOp, ComposedOp
        n = len(self._children)
        child_ops = []
        for i, child in enumerate(self._children):
            # unsqueeze to add the stack dim, then pad to fill the full stack size
            unsq = UnsqueezeOp(child.shape, self.dim)
            ndim = len(self._shape)
            pad_spec = [0] * (2 * ndim)
            d_rev = ndim - 1 - self.dim
            pad_spec[2 * d_rev] = i
            pad_spec[2 * d_rev + 1] = n - i - 1
            pad = PadOp(unsq.output_shape, pad_spec)
            embed_op = ComposedOp(pad, unsq)
            child_ops.append(weights @ embed_op)
        return (0, child_ops)

    def to_string(self, *children_str: str) -> str:
        return f"stack([{', '.join(children_str)}], dim={self.dim})"
