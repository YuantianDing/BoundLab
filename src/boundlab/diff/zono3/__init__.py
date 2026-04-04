r"""Triple-Zonotope-Based Abstract Interpretation for Differential Verification

This module provides zonotope transformations for computing over-approximations
of the difference ``f₁(x) − f₂(x)`` between two structurally identical networks,
achieving tighter bounds than verifying each network independently.

The interpreter operates on **triples** :class:`~boundlab.diff.expr.DiffExpr3`
``(x, y, d)`` where:

- ``x``: expression tracking network 1's output zonotope,
- ``y``: expression tracking network 2's output zonotope,
- ``d``: expression tracking the *difference* ``f₁(x) − f₂(x)``.

Affine operations (addition, scalar multiplication, linear layers, shape ops)
are handled directly: the bias cancels in the diff component, and weight
matrices are applied to all three components (without bias for ``d``).

Non-linear operations (ReLU, …) use specialised differential linearisers
derived from VeryDiff (Teuber et al., 2024).

Examples
--------
Build a :class:`~boundlab.diff.expr.DiffExpr3` and propagate it through a model:

>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> from boundlab.diff.expr import DiffExpr3
>>> from boundlab.diff.zono3 import interpret
>>> model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
>>> op = interpret(model)
>>> x = expr.ConstVal(torch.zeros(4)) + expr.LpEpsilon([4])
>>> y = expr.ConstVal(torch.ones(4)) + expr.LpEpsilon([4])
>>> d = x - y
>>> out = op(DiffExpr3(x, y, d))
>>> out.diff.ub().shape, out.diff.lb().shape
(torch.Size([3]), torch.Size([3]))
"""

import torch

from boundlab.expr._core import Expr
from boundlab.expr._affine import ConstVal
from boundlab.expr._var import LpEpsilon
from boundlab.interp import Interpreter
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.zono import ZonoBounds, interpret as std_interpret


# =====================================================================
# Expression builder
# =====================================================================

def _bounds_to_expr(bounds: ZonoBounds, inputs: list) -> Expr:
    """Build an :class:`~boundlab.expr.Expr` from *bounds* and its input expressions."""
    expr_sum = None
    for w, e in zip(bounds.input_weights, inputs):
        if isinstance(w, int) and w == 0:
            continue
        if isinstance(w, torch.Tensor) and not w.any():
            continue
        term = w * e
        expr_sum = term if expr_sum is None else expr_sum + term

    result = (ConstVal(bounds.bias) if expr_sum is None else expr_sum + bounds.bias)

    if bounds.error_coeffs is not None:
        new_eps = LpEpsilon(bounds.error_coeffs.input_shape)
        result = result + bounds.error_coeffs(new_eps)

    return result


# =====================================================================
# Interpreter
# =====================================================================

interpret = Interpreter[Expr | DiffExpr2 | DiffExpr3](std_interpret)
"""Differential-verification interpreter.

Feed it a :class:`~boundlab.diff.expr.DiffExpr3` ``(x, y, d)`` where *x* and
*y* are the two networks' zonotope expressions and *d* over-approximates their
difference, or a plain :class:`~boundlab.expr.Expr` for standard zonotope
interpretation.

Examples
--------
Differential mode (:class:`~boundlab.diff.expr.DiffExpr3` input):

>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> from boundlab.diff.expr import DiffExpr3
>>> from boundlab.diff.zono3 import interpret
>>> model = nn.Linear(4, 3)
>>> op = interpret(model)
>>> x = expr.ConstVal(torch.randn(4)) + expr.LpEpsilon([4])
>>> y = expr.ConstVal(torch.randn(4)) + expr.LpEpsilon([4])
>>> out = op(DiffExpr3(x, y, x - y))
>>> out.diff.ub().shape
torch.Size([3])

Fallback mode (plain :class:`~boundlab.expr.Expr`) matches standard zonotope
interpretation:

>>> z = expr.ConstVal(torch.randn(4)) + expr.LpEpsilon([4])
>>> z_out = op(z)
>>> z_out.ub().shape
torch.Size([3])
"""


# =====================================================================
# Lineariser registration
# =====================================================================

def _register_linearizer(name: str):
    """Register a differential lineariser for a non-linear activation.

    The decorated function receives ``(x, y, d)`` — the unpacked components of a
    :class:`~boundlab.diff.expr.DiffExpr3` — and returns a **triple**
    ``(x_bounds, y_bounds, d_bounds)`` of :class:`~boundlab.zono.ZonoBounds`.

    ``d_bounds.input_weights`` must have three entries ``[w_x, w_y, w_d]``
    (use the integer ``0`` to skip a term).

    For :class:`~boundlab.diff.expr.DiffExpr2` inputs the standard handler is
    applied independently to each component. Plain :class:`~boundlab.expr.Expr`
    inputs fall back to the standard zonotope handler.
    """
    def decorator(linearizer):
        std_handler = std_interpret[name]

        def handler(t: Expr | DiffExpr2 | DiffExpr3) -> Expr | DiffExpr2 | DiffExpr3:
            if isinstance(t, DiffExpr3):
                x, y, d = t.x, t.y, t.diff
                x_bounds, y_bounds, d_bounds = linearizer(x, y, d)
                return DiffExpr3(
                    _bounds_to_expr(x_bounds, [x]),
                    _bounds_to_expr(y_bounds, [y]),
                    _bounds_to_expr(d_bounds, [x, y, d]),
                )
            if isinstance(t, DiffExpr2):
                x, y = t.x, t.y, 
                d = x - y
                x_bounds, y_bounds, d_bounds = linearizer(x, y, d)
                return DiffExpr3(
                    _bounds_to_expr(x_bounds, [x]),
                    _bounds_to_expr(y_bounds, [y]),
                    _bounds_to_expr(d_bounds, [x, y, d]),
                )
            return std_handler(t)

        interpret[name] = handler
        return linearizer
    return decorator


# =====================================================================
# Activation modules (imported last so helpers are already defined)
# =====================================================================

from . import relu as _relu  # noqa: E402  — registers "relu"

interpret["ReLU"] = lambda _, x: interpret.dispatcher["relu"](x)

# diff_pair: converts paired tensors to DiffExpr2 during interpretation
from boundlab.diff.op import diff_pair_handler  # noqa: E402
interpret["diff_pair"] = diff_pair_handler

def _difflinear_handler(mod, x):
    w = DiffExpr2(ConstVal(mod.fc1.weight.detach()), ConstVal(mod.fc2.weight.detach())).transpose(0, 1)
    y = x @ w
    if mod.fc1.bias is None:
        return y
    b1 = ConstVal(mod.fc1.bias.detach())
    b2 = ConstVal(mod.fc2.bias.detach())
    # Expr arithmetic does not broadcast automatically; align bias rank to output rank.
    while len(b1.shape) < len(y.shape):
        b1 = b1.unsqueeze(0)
        b2 = b2.unsqueeze(0)
    return y + DiffExpr2(b1, b2)


interpret |= {"DiffLinear": _difflinear_handler}

from .relu import relu_linearizer  # noqa: E402, F401

__all__ = [
    "interpret",
    "relu_linearizer",
    "_register_linearizer",
]
