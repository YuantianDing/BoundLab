r"""Core Expression Classes

This module defines the base expression class and flags used throughout
the expression framework.
"""

import itertools
from typing import Literal, TYPE_CHECKING, Union

import torch
import enum


class ExprFlags(enum.Flag):
    """Flags indicating expression properties for optimization."""
    NONE = 0
    SYMMETRIC_TO_0 = enum.auto()
    """The feasible region is symmetric about zero."""
    PRINT_FUSE = enum.auto()
    """Expression should be fused with parent when printing."""
    IS_CONST = enum.auto()
    """Expression represents a constant value."""

_EXPR_ID_COUNTER = itertools.count()

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
        self.id = next(_EXPR_ID_COUNTER)
        self.flags = flags

    def with_children(self, *new_children: "Expr") -> "Expr":
        """Return a new expression with the same type and flags but new children."""
        raise NotImplementedError(f"The :code:`with_children` method is not implemented for {self.__class__.__name__}.")

    @property
    def shape(self) -> torch.Size:
        """The shape of the output(s) produced by this expression."""
        raise NotImplementedError(f"The :code:`shape` property is not implemented for {self.__class__.__name__}.")

    @property
    def children(self) -> tuple["Expr", ...]:
        """The child expressions that serve as inputs to this expression."""
        raise NotImplementedError(f"The :code:`children` property is not implemented for {self.__class__.__name__}.")

    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=", "=="] = "==") -> Union[tuple[Union[torch.Tensor, int], ...], None]:
        r"""Perform backward-mode bound propagation through this expression.

        This method computes a linear relaxation by propagating weights from
        the output back to the inputs. Given an expression $f(x_1, \ldots, x_n)$
        and output weights $\mathbf{w}$, backward propagation derives:

        $$\mathbf{w}^\top f(x_1, \ldots, x_n) \;\square\; b + \sum_i \mathbf{w}_i^\top x_i$$

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
            A list of tensors, $(b, \mathbf{w}_1, \ldots, \mathbf{w}_n)$, representing the propagated linear form
            $b + \sum_i \mathbf{w}_i^\top x_i$, or ``None`` if the expression
            cannot contribute to the bound in the specified mode, e.g. ``"=="``.
        """
        raise NotImplementedError(f"The :code:`backward` method is not implemented for {self.__class__.__name__}.")

    def to_string(self, *children_str: str) -> str:
        """Return string representation with child strings substituted."""
        return f"{self.__class__.__name__}({', '.join(children_str)})"

    def __str__(self):
        return "bl.Expr {\n" + expr_pretty_print(self, indent=4) + "\n}"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, flags={self.flags})"
    
    def __add__(self, other):
        from boundlab.expr._base import add
        if isinstance(other, (Expr, torch.Tensor)):
            return add(self, other)
        return NotImplemented
    
    def __mul__(self, other):
        from boundlab.expr._linear import linear_op
        if isinstance(other, torch.Tensor):
            return linear_op(lambda x: x * other)(self)
        return NotImplemented

    def __rmul__(self, other):
        from boundlab.expr._linear import linear_op
        if isinstance(other, torch.Tensor):
            return linear_op(lambda x: other * x)(self)
        return NotImplemented
    
    def __matmul__(self, other):
        from boundlab.expr._linear import linear_op
        if isinstance(other, torch.Tensor):
            return linear_op(lambda x: x @ other)(self)
        return NotImplemented
    
    def __rmatmul__(self, other):
        from boundlab.expr._linear import linear_op
        if isinstance(other, torch.Tensor):
            return linear_op(lambda x: other @ x)(self)
        return NotImplemented

    def reshape(self, *shape) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.reshape(*shape))(self)
    
    def permute(self, *dims) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.permute(*dims))(self)
    
    def transpose(self, dim0, dim1) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.transpose(dim0, dim1))(self)
    
    def flatten(self, start_dim=0, end_dim=-1) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.flatten(start_dim, end_dim))(self)
    
    def unflatten(self, dim, sizes) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.unflatten(dim, sizes))(self)
    
    def squeeze(self, dim=None) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.squeeze(dim))(self)
    
    def unsqueeze(self, dim) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.unsqueeze(dim))(self)
    
    def narrow(self, dim, start, length) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.narrow(dim, start, length))(self)
    
    def expand(self, *sizes) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.expand(*sizes))(self)
    
    def repeat(self, *sizes) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.repeat(*sizes))(self)
    
    def tile(self, *sizes) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.tile(*sizes))(self)
    
    def flip(self, dims) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.flip(dims))(self)
    
    def roll(self, shifts, dims) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.roll(shifts, dims))(self)
    
    def diag(self, diagonal=0) -> "Expr":
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x.diag(diagonal))(self)

    def __getitem__(self, indices):
        from boundlab.expr._linear import linear_op
        return linear_op(lambda x: x[indices])(self)

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

