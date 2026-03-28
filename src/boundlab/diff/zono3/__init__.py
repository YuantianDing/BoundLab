r"""Triple-Zonotope-Based Abstract Interpretation for Differential Verification

This module provides zonotope transformations for computing over-approximations
of the difference ``f₁(x) − f₂(x)`` between two structurally identical networks,
achieving tighter bounds than verifying each network independently.

The interpreter operates on **triples** ``(x, y, d)`` where:

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
Build a triple ``(x, y, d)`` and propagate it through a model:

>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> from boundlab.diff.zono3 import interpret
>>> model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
>>> op = interpret(model)
>>> x = expr.ConstVal(torch.zeros(4)) + expr.LpEpsilon([4])
>>> y = expr.ConstVal(torch.ones(4)) + expr.LpEpsilon([4])
>>> d = x - y
>>> _, _, d_out = op((x, y, d))
>>> d_ub, d_lb = d_out.ublb()
>>> d_ub.shape, d_lb.shape
(torch.Size([3]), torch.Size([3]))
"""

from typing import Callable

import torch

from boundlab.expr._core import Expr
from boundlab.expr._affine import ConstVal
from boundlab.expr._var import LpEpsilon
from boundlab.interp import Interpreter, _AFFINE_DISPATCHER
from boundlab.utils import Triple
from boundlab.zono import ZonoBounds, interpret as std_interpret


# =====================================================================
# Helpers
# =====================================================================

def _is_triple(x) -> bool:
    """Check if *x* is a ``Triple[Expr]``."""
    return isinstance(x, tuple) and len(x) == 3


def _with_std_fallback(diff_fn: Callable, std_fn: Callable) -> Callable:
    """Wrap *diff_fn* to fall back to *std_fn* when no argument is a triple."""
    def handler(*args, **kwargs):
        if any(_is_triple(a) for a in args):
            return diff_fn(*args, **kwargs)
        return std_fn(*args, **kwargs)
    return handler


# =====================================================================
# Affine handlers for triples
# =====================================================================

def _triple_add(a, b):
    if _is_triple(a) and _is_triple(b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    elif _is_triple(a):
        # constant added to both networks cancels in the diff
        return (a[0] + b, a[1] + b, a[2])
    else:
        return (b[0] + a, b[1] + a, b[2])


def _triple_sub(a, b):
    if _is_triple(a) and _is_triple(b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    elif _is_triple(a):
        return (a[0] - b, a[1] - b, a[2])
    else:
        # (const - x, const - y, -(x - y))
        return (a - b[0], a - b[1], -b[2])


def _triple_neg(a):
    return (-a[0], -a[1], -a[2])


def _triple_mul(a, b):
    """Element-wise multiplication by a constant (scalar / tensor / ConstVal)."""
    if _is_triple(a) and _is_triple(b):
        raise NotImplementedError(
            "Bilinear multiplication of two triples is not supported "
            "in the affine handler; register a bilinear handler instead."
        )
    if _is_triple(b):
        a, b = b, a
    # a is the triple, b is constant
    if isinstance(b, torch.Tensor):
        return (a[0] * b, a[1] * b, a[2] * b)
    else:
        return (b * a[0], b * a[1], b * a[2])


def _triple_truediv(a, b):
    if _is_triple(b):
        raise NotImplementedError("Division by a triple is not supported")
    return (a[0] / b, a[1] / b, a[2] / b)


def _triple_floordiv(a, b):
    if _is_triple(b):
        raise NotImplementedError("Division by a triple is not supported")
    return (a[0] / b, a[1] / b, a[2] / b)  # approximate


def _triple_linear_mod(mod, t):
    """``nn.Linear``: bias cancels in the diff component."""
    return (
        t[0] @ mod.weight.T + mod.bias,
        t[1] @ mod.weight.T + mod.bias,
        t[2] @ mod.weight.T,
    )


def _triple_linear_fn(t, w, b=None):
    """``F.linear``: weights arrive as ``ConstVal``."""
    bias_val = b.value if b is not None else 0
    return (
        t[0] @ w.value.T + bias_val,
        t[1] @ w.value.T + bias_val,
        t[2] @ w.value.T,
    )


def _triple_shape_op(op_name: str):
    """Return a handler that applies a shape operation to every triple element."""
    def handler(t, *args, **kwargs):
        return tuple(getattr(ti, op_name)(*args, **kwargs) for ti in t)
    return handler


_DIFF_AFFINE_DISPATCHER: dict[str, Callable] = {
    key: _with_std_fallback(diff_fn, _AFFINE_DISPATCHER[key])
    for key, diff_fn in {
        # ---- arithmetic ---------------------------------------------------
        "add":      _triple_add,
        "sub":      _triple_sub,
        "neg":      _triple_neg,
        "mul":      _triple_mul,
        "truediv":  _triple_truediv,
        "floordiv": _triple_floordiv,
        # ---- linear layers ------------------------------------------------
        "Linear":   _triple_linear_mod,
        "linear":   _triple_linear_fn,
        # ---- shape ops ----------------------------------------------------
        "reshape":    _triple_shape_op("reshape"),
        "view":       _triple_shape_op("reshape"),
        "flatten":    _triple_shape_op("flatten"),
        "permute":    _triple_shape_op("permute"),
        "transpose":  _triple_shape_op("transpose"),
        "unsqueeze":  _triple_shape_op("unsqueeze"),
        "squeeze":    _triple_shape_op("squeeze"),
        "contiguous": lambda t: t,
    }.items()
}


# =====================================================================
# Interpreter
# =====================================================================

interpret = Interpreter[Triple[Expr] | Expr](_DIFF_AFFINE_DISPATCHER, handle_affine=False)
"""Differential-verification interpreter.

Feed it triples ``(x, y, d)`` where *x* and *y* are the two networks'
zonotope expressions and *d* over-approximates their difference.

Examples
--------
Differential mode (triple input):

>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> from boundlab.diff.zono3 import interpret
>>> model = nn.Linear(4, 3)
>>> op = interpret(model)
>>> x = expr.ConstVal(torch.randn(4)) + expr.LpEpsilon([4])
>>> y = expr.ConstVal(torch.randn(4)) + expr.LpEpsilon([4])
>>> _, _, d_out = op((x, y, x - y))
>>> d_out.ub().shape
torch.Size([3])

Fallback mode (plain ``Expr``) matches standard zonotope interpretation:

>>> z = expr.ConstVal(torch.randn(4)) + expr.LpEpsilon([4])
>>> z_out = op(z)
>>> z_out.ub().shape
torch.Size([3])
"""


# =====================================================================
# Lineariser registration
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


def _register_linearizer(name: str):
    """Register a differential lineariser for a non-linear activation.

    The decorated function receives ``(x, y, d)`` and returns a **triple**
    ``(x_bounds, y_bounds, d_bounds)`` of :class:`~boundlab.zono.ZonoBounds`.

    ``d_bounds.input_weights`` must have three entries ``[w_x, w_y, w_d]``
    (use the integer ``0`` to skip a term).
    """
    def decorator(linearizer):
        std_handler = std_interpret.dispatcher[name]

        def handler(t: Triple[Expr] | Expr) -> Triple[Expr] | Expr:
            if not _is_triple(t):
                return std_handler(t)

            x, y, d = t
            x_bounds, y_bounds, d_bounds = linearizer(x, y, d)

            return (
                _bounds_to_expr(x_bounds, [x]),
                _bounds_to_expr(y_bounds, [y]),
                _bounds_to_expr(d_bounds, [x, y, d]),
            )

        interpret.dispatcher[name] = handler
        return linearizer
    return decorator


# =====================================================================
# Activation modules (imported last so helpers are already defined)
# =====================================================================

from . import relu as _relu  # noqa: E402  — registers "relu"

interpret.dispatcher["ReLU"] = lambda _, x: interpret.dispatcher["relu"](x)

from .relu import relu_linearizer  # noqa: E402, F401

__all__ = [
    "interpret",
    "relu_linearizer",
]
