"""Softmax2 handler for polytope abstract interpretation.

Defines

.. math::

   \mathrm{softmax2}(x, y) = \frac{x}{1 + x\,\exp(y)}
"""

from __future__ import annotations

from boundlab.expr._core import Expr


def softmax2_handler(x: Expr, y: Expr) -> Expr:
    """Over-approximate ``x / (1 + x * exp(y))`` using polytope primitives."""
    assert x.shape == y.shape, f"softmax2 expects matching shapes, got {x.shape} vs {y.shape}"

    from . import interpret

    exp_y = interpret["Exp"](y)
    denom = 1.0 + interpret["Mul"](x, exp_y)
    inv = interpret["Reciprocal"](denom)
    return interpret["Mul"](x, inv)

