from __future__ import annotations

r"""Linear Operations for Expressions

This module provides ``AffineSum``, a fused expression class that represents
a sum of EinsumOp-weighted children: Σ_i op_i(child_i).
It replaces separate linear-sequence and add-node structures from the
previous design.
"""

from typing import Literal

import torch

from boundlab.expr._core import Expr, ExprFlags
from boundlab.linearop import EinsumOp, LinearOp

class AffineSum(Expr):
    r"""An expression representing a sum of linear operations applied to children.

    Represents :math:`\sum_i \mathrm{op}_i(x_i)` where each :math:`\mathrm{op}_i`
    is a :class:`~boundlab.linearop.EinsumOp`.

    During construction, if a child is itself an :class:`AffineSum`, its pairs
    are absorbed by composing the outer op with each inner op via ``@``
    (eager contraction). This ensures the expression tree is always flat
    — no :class:`AffineSum` node ever has an :class:`AffineSum` child.

    Attributes:
        pairs: List of ``(op, child)`` tuples.
        ops: List of EinsumOp operators (convenience view).
    """

    def __init__(self, *pairs: tuple, const=None):
        """Construct an AffineSum.

        Args:
            *pairs: Sequence of ``(op, child)`` pairs where ``op`` is a
                :class:`~boundlab.linearop.EinsumOp` and ``child`` is
                an :class:`Expr` or :class:`torch.Tensor`.
        """
        super().__init__(ExprFlags.IS_AFFINE)
        self.constant = const

        # Pre-process before allocating ID so ConstVal wrappers get lower IDs.
        self.children_dict: dict[Expr, LinearOp] = {}
        for op, child in pairs:
            if isinstance(child, torch.Tensor):
                self._add_constant(op.forward(child))
            elif isinstance(child, AffineSum):
                # Distribute op through child's pairs: (op ∘ child_op_i, grandchild_i)
                if child.constant is not None:
                    self._add_constant(op.forward(child.constant))
                for grandchild, child_op in child.children_dict.items():
                    self._add_expr(op @ child_op, grandchild)
            else:
                self._add_expr(op, child)

        output_shapes = {op.output_shape for op in self.children_dict.values()}
        if self.constant is not None:
            output_shapes.add(self.constant.shape)
        assert len(output_shapes) == 1, \
            f"All ops must share the same output shape; got {output_shapes}."
        self._shape = output_shapes.pop()
        # Propagate flags
        if self.children_dict:
            if all(ExprFlags.SYMMETRIC_TO_0 in child.flags for child in self.children_dict.keys()):
                self.flags |= ExprFlags.SYMMETRIC_TO_0
            if all(child.flags & ExprFlags.IS_CONST for child in self.children_dict.keys()):
                self.flags |= ExprFlags.IS_CONST
        else:
            self.flags |= ExprFlags.IS_CONST

    def _add_constant(self, const: torch.Tensor):
        """Accumulate a constant term into this AffineSum."""
        if const is not None:
            self.constant = self.constant + const if self.constant is not None else const
    def _add_expr(self, op: LinearOp, child: Expr):
        """Accumulate an ``(op, child)`` contribution into this AffineSum."""
        if child in self.children_dict:
            # If child already exists, compose the ops: old_op + op
            old_op = self.children_dict[child]
            new_op = old_op + op
            self.children_dict[child] = new_op
        else:
            self.children_dict[child] = op
    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def children(self) -> tuple[Expr, ...]:
        return tuple(self.children_dict.keys())

    def with_children(self, *new_children: Expr) -> "AffineSum":
        """Return a new AffineSum with the same ops but new children."""
        return AffineSum(*zip(self.children_dict.values(), new_children))

    def backward(self, weights, direction: Literal[">=", "<=", "=="]) \
            -> tuple:
        """Propagate weights backward: each child gets weights ∘ op_i.

        Args:
            weights: A :class:`~boundlab.linearop.EinsumOp` accumulated weight.
            direction: Bound direction (unused — Linear is always linear).

        Returns:
            ``(bias, [weights @ op_i for op_i in self.children_dict.values()])``.
        """
        bias = 0
        if self.constant is not None:
            bias = weights.forward(self.constant)
        return (bias, [weights @ op for op in self.children_dict.values()])

    def to_string(self, *children_str: str) -> str:
        parts = [f"{op}({cs})" for op, cs in zip(self.children_dict.values(), children_str)]
        return " + ".join(parts)
