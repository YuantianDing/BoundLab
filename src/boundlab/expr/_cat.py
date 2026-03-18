r"""Concatenation and Stacking Operations

This module provides expressions for concatenating and stacking
child expressions along specified dimensions.
"""

from typing import Literal

import torch

from boundlab.expr._core import Expr, ExprFlags


class Cat(Expr):
    """Expression for concatenating child expressions along a dimension.

    The backward pass produces an embed LinearOp per child that zero-pads
    the child's contribution into the full cat output shape. The VJP of
    F.pad (narrow) is computed automatically.
    """

    def __init__(self, *children: Expr, dim: int = 0):
        super().__init__(ExprFlags.IS_AFFINE)
        assert len(children) >= 1, "Cat requires at least one child."
        assert all(
            c.shape[:dim] == children[0].shape[:dim] and c.shape[dim + 1:] == children[0].shape[dim + 1:]
            for c in children
        ), "All children must have matching shapes except along the concatenation dimension."
        self._children = tuple(children)
        self.dim = dim

        cat_size = sum(c.shape[dim] for c in children)
        s = list(children[0].shape)
        s[dim] = cat_size
        self._shape = torch.Size(s)

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def children(self) -> tuple[Expr, ...]:
        return self._children

    def with_children(self, *new_children: Expr) -> "Cat":
        return Cat(*new_children, dim=self.dim)

    def backward(self, weights, direction: Literal[">=", "<=", "=="] = "=="):  # noqa: ARG002
        """Propagate weights to each child via zero-padding embed ops.

        Args:
            weights: A :class:`~boundlab.linearop.EinsumOp` accumulated weight.
            direction: Unused (Cat is always linear).

        Returns:
            ``(0, [child_weight_0, child_weight_1, ...])``
        """
        from boundlab.linearop import PadOp
        child_ops = []
        offset = 0
        cat_size = self._shape[self.dim]
        for child in self._children:
            size = child.shape[self.dim]
            pad_before = offset
            pad_after = cat_size - offset - size
            ndim = len(child.shape)
            # F.pad spec: pairs in reverse dim order, (left, right) per dim
            pad_spec = [0] * (2 * ndim)
            pad_spec[2 * (ndim - 1 - self.dim)] = pad_before
            pad_spec[2 * (ndim - 1 - self.dim) + 1] = pad_after
            embed_op = PadOp(child.shape, pad_spec)
            child_ops.append(weights @ embed_op)
            offset += size
        return (0, child_ops)

    def to_string(self, *children_str: str) -> str:
        return f"cat([{', '.join(children_str)}], dim={self.dim})"


class Stack(Expr):
    """Expression for stacking child expressions along a new dimension.

    All children must have identical shapes. The backward pass produces
    an embed LinearOp per child that places the child at its index along
    the stacking dimension, with zeros elsewhere.
    """

    def __init__(self, *children: Expr, dim: int = 0):
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
