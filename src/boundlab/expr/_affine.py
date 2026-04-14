from __future__ import annotations

r"""Linear Operations for Expressions

This module provides ``AffineSum``, a fused expression class that represents
a sum of EinsumOp-weighted children: Σ_i op_i(child_i).
It replaces separate linear-sequence and add-node structures from the
previous design.
"""

import sys
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

    def __new__(cls, *pairs: tuple, const=None, **_kw):
        if cls is not AffineSum:
            # ConstVal (and other subclasses) construct themselves directly.
            return object.__new__(cls)
        if len(pairs) == 0 or all(isinstance(child, ConstVal) for _, child in pairs):
            # All-constant result → return a ConstVal shell;
            # AffineSum.__init__ will populate .constant, and we sync .value below.
            return object.__new__(ConstVal)
        return object.__new__(AffineSum)
    
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
            assert isinstance(child, Expr), "Tuple expressions are not supported as children of AffineSum; use multiple arguments instead."
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
        parts = [f"{op}{cs}" for op, cs in zip(self.children_dict.values(), children_str)]
        if self.constant is not None:
            parts.append(f"<Const>")
        return " + ".join(parts)
    
    def simplify_ops_(self):
        self.children_dict = {child: op.einsum_op() for child, op in self.children_dict.items()}

    def symmetric_decompose(self) -> tuple[Expr | Literal[0], Expr | Literal[0]]:
        """Decompose this AffineSum into a constant part and a zero-constant AffineSum."""
        if self.constant is None:
            return self, None
        const_part = ConstVal(self.constant)
        if not self.children_dict:
            return const_part, None
        non_const_part = AffineSum(*zip(self.children_dict.values(), self.children_dict.keys()))
        return const_part, non_const_part

class ConstVal(AffineSum):
    """Expression representing a constant tensor value.

    Implemented as an AffineSum with no children and only a constant term.
    When used as a child of another AffineSum, the constant is automatically
    absorbed via eager contraction.
    """

    def __init__(self, value=None, name=None, *_pairs, const=None):
        # Three call patterns:
        # 1. ConstVal(tensor[, name])         — direct construction
        # 2. ConstVal(const=tensor)           — from AffineSum(const=x), no pairs
        # 3. ConstVal((op,ch), ..., const=x)  — from AffineSum(*pairs, const=x),
        #                                       all-ConstVal children; value is
        #                                       the first pair tuple, _pairs are the rest
        if isinstance(value, tuple) or _pairs:
            # Pattern 3: routed from AffineSum with pairs
            all_pairs = ((value,) + _pairs) if value is not None else _pairs
            AffineSum.__init__(self, *all_pairs, const=const)
        else:
            # Pattern 1 or 2
            actual = value if const is None else const
            AffineSum.__init__(self, const=actual)
        self.value = self.constant
        self.name = name

    def to_string(self) -> str:
        if self.name is not None:
            return f"#const {self.name}"
        return f"#const <{self.id:X}>"
    
    def get_const(self):
        if self.value is None:
            return 0
        else:
            return self.value
        
    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(self.get_const() + other)
        elif isinstance(other, ConstVal):
            return ConstVal(self.get_const() + other.get_const())
        return super().__add__(other)
    
    def __radd__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(other + self.get_const())
        if isinstance(other, ConstVal):
            return ConstVal(other.get_const() + self.get_const())
        return super().__radd__(other)
    
    def __neg__(self):
        return ConstVal(-self.get_const())
    
    def __sub__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(self.get_const() - other)
        elif isinstance(other, ConstVal):
            return ConstVal(self.get_const() - other.get_const())
        return super().__sub__(other)
    
    def __rsub__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(other - self.get_const())
        elif isinstance(other, ConstVal):
            return ConstVal(other.get_const() - self.get_const())
        return super().__rsub__(other)
    
    def __truediv__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(self.get_const() / other)
        elif isinstance(other, ConstVal):
            return ConstVal(self.get_const() / other.get_const())
        return super().__truediv__(other)
    
    def __rtruediv__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(other / self.get_const())
        elif isinstance(other, ConstVal):
            return ConstVal(other.get_const() / self.get_const())
        return super().__rtruediv__(other)
    
    def __abs__(self):
        return ConstVal(abs(self.get_const()))
        
    def __mul__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(self.get_const() * other)
        elif isinstance(other, ConstVal):
            return ConstVal(self.get_const() * other.get_const())
        return super().__mul__(other)
    
    def __rmul__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(other * self.get_const())
        if isinstance(other, ConstVal):
            return ConstVal(other.get_const() * self.get_const())
        return super().__rmul__(other)

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(self.get_const() @ other)
        elif isinstance(other, ConstVal):
            return ConstVal(self.get_const() @ other.get_const())
        return super().__matmul__(other)
    
    def __rmatmul__(self, other):
        if isinstance(other, torch.Tensor):
            return ConstVal(other @ self.get_const())
        elif isinstance(other, ConstVal):
            return ConstVal(other.get_const() @ self.get_const())
        return super().__rmatmul__(other)
    
    def _apply_op(self, op):
        return ConstVal(op.forward(self.get_const()))