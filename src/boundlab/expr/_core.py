r"""Core Expression Classes

This module defines the base expression class and flags used throughout
the expression framework.
"""

from typing import Literal, TYPE_CHECKING

import torch
import enum
from atomicx import AtomicInt
from boundlab.utils import eye_of

if TYPE_CHECKING:
    from boundlab.expr._base import Add, SubTensor
    from boundlab.expr._mul import ConstTensorDot


class ExprFlags(enum.Flag):
    """Flags indicating expression properties for optimization."""
    NONE = 0
    SYMMETRIC_TO_0 = enum.auto()
    """The feasible region is symmetric about zero."""
    PRINT_FUSE = enum.auto()
    """Expression should be fused with parent when printing."""
    IS_CONST = enum.auto()
    """Expression represents a constant value."""
    IS_CONST_MULTIPLICATIVE = enum.auto()
    """Expression is a constant multiplicative transformation."""

_EXPR_ID_COUNTER = AtomicInt()

class Expr:
    """Abstract base class for all symbolic expressions in BoundLab.

    Each expression represents a node in a directed acyclic graph (DAG) of
    computations. Expressions are immutable after construction, and each
    instance is assigned a unique time-ordered UUID to enable deterministic
    topological ordering during bound propagation.

    Subclasses must implement :attr:`shape`, :attr:`children`, and
    :meth:`backward` to define the expression's semantics.
    """

    id: int
    """Unique identifier for the expression, used for topological sorting."""
    flags: ExprFlags
    """Flags indicating expression properties for optimization."""

    def __init__(self, flags: ExprFlags = ExprFlags.NONE):
        self.id = _EXPR_ID_COUNTER.inc()
        self.flags = flags

    @property
    def shape(self) -> torch.Size:
        """The shape of the output(s) produced by this expression."""
        raise NotImplementedError(f"The :code:`shape` property is not implemented for {self.__class__.__name__}.")

    @property
    def children(self) -> tuple["Expr", ...]:
        """The child expressions that serve as inputs to this expression."""
        raise NotImplementedError(f"The :code:`children` property is not implemented for {self.__class__.__name__}.")

    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=", "=="] = "==") -> "Expr | None":
        r"""Perform backward-mode bound propagation through this expression.

        This method computes a linear relaxation by propagating weights from
        the output back to the inputs. Given an expression $f(x_1, \ldots, x_n)$
        and output weights $\mathbf{w}$, backward propagation derives:

        $$\mathbf{w}^\top f(x_1, \ldots, x_n) \;\square\; \sum_i \mathbf{w}_i^\top x_i + b$$

        where $\square$ is determined by the mode parameter:

        - ``">="`` computes a lower bound relaxation
        - ``"<="`` computes an upper bound relaxation
        - ``"=="`` computes an exact linear transformation (when applicable)

        Args:
            weights: Weight tensor to propagate, with shape compatible with
                the expression's output shape.
            mode: Bound direction. One of ``">="`` (lower), ``"<="`` (upper),
                or ``"=="`` (exact).

        Returns:
            An expression representing the propagated linear form
            $\sum_i \mathbf{w}_i^\top x_i + b$, or ``None`` if the expression
            cannot contribute to the bound in the specified mode.
        """
        raise NotImplementedError(f"The :code:`backward` method is not implemented for {self.__class__.__name__}.")

    def simplify(self) -> "Expr":
        """Apply algebraic simplifications to reduce the expression DAG.

        Returns:
            A simplified expression that is semantically equivalent to this one.
        """
        return self

    def add_simplify(self, other: "Expr") -> "Expr | None":
        """Attempt to merge this expression with another under addition.

        This method enables combining like terms when constructing sums,
        reducing the number of nodes in the expression DAG.

        Args:
            other: Another expression to potentially merge with this one.

        Returns:
            A merged expression if the two can be combined, or ``None``
            if no simplification is applicable.
        """
        return None

    def to_string(self, *children_str: str) -> str:
        """Return string representation with child strings substituted."""
        return f"{self.__class__.__name__}({', '.join(children_str)})"

    def __str__(self):
        return "bl.Expr {\n" + expr_pretty_print(self, indent=4) + "\n}"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, flags={self.flags})"

    def __add__(self, other) -> "Add":
        from boundlab.expr._base import add
        return add(self, other)

    def __radd__(self, other) -> "Add":
        from boundlab.expr._base import add
        return add(other, self)

    def __mul__(self, other) -> "Expr":
        if isinstance(other, (int, float)):
            from boundlab.expr._mul import ConstTensorDot
            return ConstTensorDot(torch.tensor(other), self, dims=0)
        raise NotImplementedError(f"Multiplication is only supported between an expression and a scalar constant. Got {self.__class__.__name__} * {other.__class__.__name__}.")

    def __rmul__(self, other) -> "Expr":
        if isinstance(other, (int, float)):
            from boundlab.expr._mul import ConstTensorDot
            return ConstTensorDot(torch.tensor(other), self, dims=0)
        raise NotImplementedError(f"Multiplication is only supported between an expression and a scalar constant. Got {other.__class__.__name__} * {self.__class__.__name__}.")

    def __getitem__(self, indices: int | slice | tuple[int | slice, ...]) -> "SubTensor":
        from boundlab.expr._base import SubTensor
        if isinstance(indices, (int, slice)):
            indices = (indices,)
        return SubTensor(self, indices)

    def get_all_leave_ids(self) -> set[int]:
        """Recursively collect all :attr:`id` in leaf expressions in the DAG."""
        if not self.children:
            return {self.id}
        leaves = set()
        for child in self.children:
            leaves.update(child.get_all_leave_ids())
        return leaves

    def ub(self) -> torch.Tensor:
        """Compute an upper bound for this expression."""
        from boundlab import prop
        return prop.ub(self)

    def lb(self) -> torch.Tensor:
        """Compute a lower bound for this expression."""
        from boundlab import prop
        return prop.lb(self)

    def ublb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both an upper bound and a lower bound for this expression."""
        from boundlab import prop
        return prop.ublb(self)

    def backward_eye(self, mode: Literal[">=", "<=", "=="] = "==") -> "Expr | None":
        """Convenience method for backward propagation with identity weights."""
        output_shape = self.shape
        eye_weights = eye_of(output_shape)
        return self.backward(eye_weights, mode=mode)


def expr_pretty_print(expr: Expr, indent: int = 0) -> str:
    """Pretty print an expression in SSA form.

    Args:
        expr: The expression to print.
        indent: Number of spaces to indent each line.

    Returns:
        A string representation of the expression DAG.
    """
    visited = list()

    def dfs(e: Expr):
        if e in visited:
            return
        visited.append(e)
        for child in e.children:
            dfs(child)
    dfs(expr)

    visited.reverse()

    # Count references to each expression
    ref_count = {e: 0 for e in visited}
    for e in visited:
        for child in e.children:
            ref_count[child] += 1

    # Expressions that can be fused (PRINT_FUSE flag and only one reference)
    def can_fuse(e: Expr) -> bool:
        return (ExprFlags.PRINT_FUSE in e.flags) and ref_count[e] <= 1

    # Build string representation with fusing
    expr_to_str = {}

    def get_expr_str(e: Expr) -> str:
        if e in expr_to_str:
            return expr_to_str[e]
        children_strs = []
        for child in e.children:
            if can_fuse(child):
                children_strs.append(get_expr_str(child))
            else:
                children_strs.append(f'%{visited.index(child)}')
        result = e.to_string(*children_strs)
        expr_to_str[e] = result
        return result

    output = []
    for i, e in enumerate(visited):
        # Skip expressions that will be fused into their parent
        if can_fuse(e):
            continue
        get_expr_str(e)
        output.append((" " * indent) + f"%{i} = {expr_to_str[e]}")
    return "\n".join(output)

