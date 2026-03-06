r"""Basic Expression Classes

This module defines fundamental expression types including constants,
addition, and tensor slicing operations.
"""

from typing import Literal, TYPE_CHECKING, Union, Optional

import torch

from boundlab.expr._core import Expr, ExprFlags

class ConstVal(Expr):
    r"""A leaf expression representing a constant tensor value.

    During backward propagation, this expression contributes a bias term
    computed as $\mathbf{w}^\top \mathbf{c}$, where $\mathbf{w}$ is the
    propagated weight and $\mathbf{c}$ is the constant value.
    """
    def __init__(self, value: torch.Tensor, name: Optional[str] = None):
        super().__init__(ExprFlags.IS_CONST)
        self.value = value
        self.name = name

    @property
    def shape(self) -> torch.Size:
        return self.value.shape
    @property
    def children(self) -> tuple[()]:
        return ()

    def with_children(self) -> "ConstVal":
        """Return self (leaf expressions have no children to replace)."""
        return self

    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=","=="]="==") -> tuple[torch.Tensor]:
        return (torch.tensordot(weights, self.value, dims=self.value.dim()),)
    
    def to_string(self) -> str:
        if self.name is not None:
            return "#const " + self.name
        return f"#const " + self.id


class Add(Expr):
    r"""An expression representing the element-wise sum of child expressions.

    All children must have identical shapes. During backward bound propagation,
    the same weight tensor is distributed to each child term.
    """
    def __init__(self, *children: Expr):
        super().__init__()
        self._children = tuple(children)
        assert all(not isinstance(child, Add) for child in children), "Nested Add expressions should be flattened. Please use the add() helper function to construct sums."
        if all(ExprFlags.SYMMETRIC_TO_0 in child.flags for child in children):
            self.flags |= ExprFlags.SYMMETRIC_TO_0
        if all(child.flags & ExprFlags.IS_CONST for child in children):
            self.flags |= ExprFlags.IS_CONST
        assert all(child.shape == children[0].shape for child in children), "Children must have the same shape."

    @property
    def shape(self) -> torch.Size:
        return self.children[0].shape
    
    @property
    def children(self) -> tuple[Expr, ...]:
        return self._children

    def with_children(self, *new_children: Expr) -> "Add":
        """Return a new Add expression with the given children."""
        return sum_exprs(*new_children)

    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=","=="]="==") -> tuple[Union[torch.Tensor, int], ...]:
        return (0, *(weights for _ in self.children))

    def to_string(self, *children_str: str) -> str:
        return " + ".join(children_str)

def sum_exprs(*children: Union[Expr, torch.Tensor]) -> Add:
    """Construct an Add expression from expressions or tensors.

    Tensors are automatically wrapped in :class:`ConstVal`, and nested
    :class:`Add` expressions are flattened into a single level.

    This function is also available as :func:`add` for convenience.
    """
    processed_children = []
    for child in children:
        if not isinstance(child, Expr) and not isinstance(child, torch.Tensor):
            processed_children.append(ConstVal(torch.tensor(child)))
        elif isinstance(child, torch.Tensor):
            processed_children.append(ConstVal(child))
        elif isinstance(child, Add):
            processed_children.extend(child.children)
        else:
            processed_children.append(child)
    return Add(*processed_children)


# Alias for backwards compatibility
add = sum_exprs
