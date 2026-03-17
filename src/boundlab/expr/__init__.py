r"""Symbolic Expression Framework for Bound Propagation

This module provides the foundational expression classes and operators for
constructing and manipulating symbolic representations of neural network
computations in BoundLab. These expressions serve as the basis for computing
sound over-approximations of reachable sets through bound propagation.

Consider a zonotope, a commonly used abstract domain for neural network
verification, expressed in the canonical form:

$$Z = c + \mathbf{G} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \in [-1, 1]^m$$

where $c \in \mathbb{R}^n$ denotes the center, $\mathbf{G} \in \mathbb{R}^{n \times m}$
is the generator matrix, and $\boldsymbol{\epsilon}$ represents the noise symbols.

A key feature of this framework is structural sharing: multiple expressions may
reference the same subexpression. When two zonotopes $Z_1$ and $Z_2$ share
common error terms, the framework preserves these dependencies, enabling
tighter bound computation through correlation tracking.

The module implements backward-mode bound propagation, which computes linear
relaxations by propagating weight matrices from outputs to inputs.
"""

# Import core classes first (no circular dependencies)
from ._core import Expr, ExprFlags, expr_pretty_print
from ._base import ConstVal
from ._linear import AffineSum
from ._var import LpEpsilon
from ._cat import Cat, Stack


def Add(*children: Expr) -> AffineSum:
    """Create an AffineSum of children with identity weights (convenience alias)."""
    from boundlab.linearop import ScalarMul
    return AffineSum(*((ScalarMul(1.0, c.shape), c) for c in children))


__all__ = [
    # Core
    "Expr",
    "ExprFlags",
    "expr_pretty_print",
    # Base expressions
    "ConstVal",
    # Linear operations
    "AffineSum",
    "Add",
    # Variable expressions
    "LpEpsilon",
    # Concatenation
    "Cat",
    "Stack",
]