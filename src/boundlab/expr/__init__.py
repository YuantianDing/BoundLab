r"""Symbolic Expression Framework for Bound Propagation

This module provides the foundational expression classes and operators for
constructing and manipulating symbolic representations of neural network
computations in BoundLab. These expressions serve as the basis for computing
sound over-approximations of reachable sets through bound propagation.

Consider a zonotope, a commonly used abstract domain for neural network
verification, expressed in the canonical form:

.. math::

   Z = c + \mathbf{G} \boldsymbol{\epsilon}, \quad
   \boldsymbol{\epsilon} \in [-1, 1]^m

where :math:`c \in \mathbb{R}^n` denotes the center,
:math:`\mathbf{G} \in \mathbb{R}^{n \times m}` is the generator matrix, and
:math:`\boldsymbol{\epsilon}` represents the noise symbols.

A key feature of this framework is structural sharing: multiple expressions may
reference the same subexpression. When two zonotopes $Z_1$ and $Z_2$ share
common error terms, the framework preserves these dependencies, enabling
tighter bound computation through correlation tracking.

The module implements backward-mode bound propagation, which computes linear
relaxations by propagating weight matrices from outputs to inputs.

Examples
--------
Build ``center + epsilon`` and query bounds:

>>> import torch
>>> import boundlab.expr as expr
>>> x = expr.ConstVal(torch.tensor([0.5, -1.0])) + expr.LpEpsilon([2])
>>> ub, lb = x.ublb()
>>> torch.allclose(ub, torch.tensor([1.5, 0.0]))
True
>>> torch.allclose(lb, torch.tensor([-0.5, -2.0]))
True
"""

# Import core classes first (no circular dependencies)
from ._core import Expr, ExprFlags, expr_pretty_print
from ._affine import AffineSum, ConstVal
from ._var import LpEpsilon
from ._cat import Cat, Stack
from ._tuple import TupleExpr, MakeTuple, GetTupleItem


def Add(*children: Expr) -> AffineSum:
    r"""Construct an affine sum with unit coefficients.

    This is a convenience alias for creating:

    .. math::

       \sum_i x_i

    where each input expression :math:`x_i` receives an identity scalar
    coefficient ``1.0``.

    Args:
        *children: Input expressions with matching shapes.

    Returns:
        An :class:`AffineSum` representing the pointwise sum of ``children``.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> a = expr.ConstVal(torch.tensor([1.0, 2.0]))
    >>> b = expr.LpEpsilon([2])
    >>> s = expr.Add(a, b)
    >>> s.shape
    torch.Size([2])
    """
    from boundlab.linearop import ScalarOp
    return AffineSum(*((ScalarOp(1.0, c.shape), c) for c in children))


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
    "TupleExpr",
    "MakeTuple",
    "GetTupleItem",
]
