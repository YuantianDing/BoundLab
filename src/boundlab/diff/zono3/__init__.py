from __future__ import annotations

from boundlab import interp
from boundlab.diff import expr
from boundlab.utils import not0

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

import dataclasses

import torch

from boundlab.expr._core import Expr
from boundlab.expr._affine import ConstVal
from boundlab.expr._var import LpEpsilon
from boundlab.interp import Interpreter  # noqa: F401
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.linearop._base import LinearOp
from boundlab.zono import ZonoBounds, interpret as std_interpret


# =====================================================================
# Expression builders
# =====================================================================

def _apply_weights(weights, inputs) -> Expr | None:
    """Return the weighted sum of inputs, skipping zero weights. Returns None if all zero."""
    result = None
    for w, e in zip(weights, inputs):
        if isinstance(w, int) and w == 0:
            continue
        term = w * e
        result = term if result is None else result + term
    return result


def _build_triple_from_dzb(
        dzb: "DiffZonoBounds",
        xs: list[Expr],
        ys: list[Expr],
        ds: list[Expr],
) -> DiffExpr3:
    """Build a :class:`~boundlab.diff.expr.DiffExpr3` from *dzb*, sharing epsilon variables.

    *xs*, *ys*, *ds* are parallel lists — one entry per input to the
    nonlinearity (length 1 for unary ops, 2 for binary, etc.).
    ``x_bounds.input_weights[i]`` is applied to ``xs[i]``, and so on.

    The fresh epsilon introduced for ``x_bounds.error_coeffs`` (``eps_x``) is
    reused verbatim in ``diff_x_error(eps_x)``, and likewise for ``eps_y``.
    This makes the diff expression track ``x_output − y_output`` **exactly**
    for neurons handled by the cases 1–8 path (no extra approximation error),
    yielding tighter bounds — especially for L2 perturbations and multi-layer
    networks where the shared epsilon structure cancels downstream.
    """
    # Build x expression; capture the fresh eps_x for reuse in diff.
    x_sum = _apply_weights(dzb.x_bounds.input_weights, xs)
    x_result = ConstVal(dzb.x_bounds.bias) if x_sum is None else x_sum + dzb.x_bounds.bias
    eps_x = None
    if dzb.x_bounds.error_coeffs is not None:
        eps_x = LpEpsilon(dzb.x_bounds.error_coeffs.input_shape)
        x_result = x_result + dzb.x_bounds.error_coeffs(eps_x)

    # Build y expression; capture the fresh eps_y for reuse in diff.
    y_sum = _apply_weights(dzb.y_bounds.input_weights, ys)
    y_result = ConstVal(dzb.y_bounds.bias) if y_sum is None else y_sum + dzb.y_bounds.bias
    eps_y = None
    if dzb.y_bounds.error_coeffs is not None:
        eps_y = LpEpsilon(dzb.y_bounds.error_coeffs.input_shape)
        y_result = y_result + dzb.y_bounds.error_coeffs(eps_y)

    # Build diff expression, reusing eps_x and eps_y.
    d_result = ConstVal(dzb.diff_bounds.bias)

    if dzb.diff_x_weights != 0:
        s = _apply_weights(dzb.diff_x_weights, xs)
        if s is not None:
            d_result = d_result + s

    if dzb.diff_y_weights != 0:
        s = _apply_weights(dzb.diff_y_weights, ys)
        if s is not None:
            d_result = d_result + s

    d_in = _apply_weights(dzb.diff_bounds.input_weights, ds)
    if d_in is not None:
        d_result = d_result + d_in

    # Shared errors: same eps variables as x_result and y_result.
    if eps_x is not None and not0(dzb.diff_x_error):
        d_result = d_result + dzb.diff_x_error(eps_x)
    if eps_y is not None and not0(dzb.diff_y_error):
        d_result = d_result + dzb.diff_y_error(eps_y)

    # Fresh diff-only error (e.g. case-9 triangle relaxation on d directly).
    if dzb.diff_bounds.error_coeffs is not None:
        eps_d = LpEpsilon(dzb.diff_bounds.error_coeffs.input_shape)
        d_result = d_result + dzb.diff_bounds.error_coeffs(eps_d)

    return DiffExpr3(x_result, y_result, d_result)


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


@dataclasses.dataclass
class DiffZonoBounds:
    x_bounds: ZonoBounds
    y_bounds: ZonoBounds

    diff_bounds: ZonoBounds

    diff_x_error: LinearOp
    diff_x_weights: list[torch.Tensor | 0] | 0

    diff_y_error: LinearOp
    diff_y_weights: list[torch.Tensor | 0] | 0


# =====================================================================
# Lineariser registration
# =====================================================================

def _register_linearizer(name: str):
    """Register a differential lineariser for a non-linear activation.

    The decorated function receives ``(xs, ys, ds)`` — three parallel lists of
    :class:`~boundlab.expr.Expr`, one entry per input to the nonlinearity —
    and returns a :class:`DiffZonoBounds`.

    For unary activations (relu, tanh, …) each list has length 1.  For binary
    operations each list has length 2, with ``xs[i]`` / ``ys[i]`` / ``ds[i]``
    being the *i*-th input's x-network, y-network, and diff components
    respectively.

    ``diff_bounds.input_weights[i]`` is the weight applied to ``ds[i]``;
    ``diff_x_weights[i]`` / ``diff_y_weights[i]`` are the weights applied to
    ``xs[i]`` / ``ys[i]``.  ``diff_x_error`` / ``diff_y_error`` are applied to
    the **same** epsilon variables introduced for ``x_bounds`` / ``y_bounds``,
    enabling exact diff tracking for cases where no fresh error is needed.

    All inputs must be :class:`~boundlab.diff.expr.DiffExpr3` or
    :class:`~boundlab.diff.expr.DiffExpr2`; if none are, the call falls back to
    the standard zonotope handler.  :class:`~boundlab.diff.expr.DiffExpr2`
    inputs have their diff synthesised as ``x − y``.
    """

    def decorator(linearizer):
        def handler(*args):
            if not any(isinstance(a, (DiffExpr3, DiffExpr2)) for a in args):
                return NotImplemented
            xs, ys, ds = [], [], []
            for a in args:
                if isinstance(a, DiffExpr3):
                    xs.append(a.x);
                    ys.append(a.y);
                    ds.append(a.diff)
                elif isinstance(a, DiffExpr2):
                    xs.append(a.x);
                    ys.append(a.y);
                    ds.append(a.x - a.y)
                else:
                    xs.append(a);
                    ys.append(a);
                    ds.append(expr.ConstVal(None))  # constant: diff is 0
            return _build_triple_from_dzb(linearizer(xs, ys, ds), xs, ys, ds)

        interpret[name] = handler
        return linearizer

    return decorator


# =====================================================================
# Activation modules (imported last so helpers are already defined)
# =====================================================================

from . import relu as _relu  # noqa: E402  — registers "relu"
from . import tanh as _tanh  # noqa: E402  — registers "tanh"
from . import exp as _exp  # noqa: E402  — registers "exp"
from . import reciprocal as _reciprocal  # noqa: E402  — registers "Reciprocal"
from . import heaviside as _heaviside  # noqa: E402  — registers "HeavisidePruning"

# ONNX activation op names
interpret["Relu"] = interpret["relu"]
interpret["Tanh"] = interpret["tanh"]

# diff_pair: converts paired tensors (from boundlab::diff_pair ONNX nodes) to DiffExpr2
from boundlab.diff.op import diff_pair_handler  # noqa: E402

interpret["DiffPair"] = diff_pair_handler

# Bilinear handlers (differential mul/matmul)
from .bilinear import diff_mul_handler, diff_matmul_handler  # noqa: E402

def onnx_boardcasted(fn):
    return lambda X, Y, *args, **kwargs: fn(*interp._onnx_broadcast(X, Y), *args, **kwargs)

interpret["Mul"] = onnx_boardcasted(diff_mul_handler)
interpret["MatMul"] = diff_matmul_handler
interpret["Div"] = onnx_boardcasted(lambda a, b: diff_mul_handler(a, interpret["Reciprocal"](b)))

# Softmax: both call_module (nn.Softmax) and ATen lowered (_softmax.default)
from .softmax import diff_softmax_handler  # noqa: E402

# interpret["softmax"] = diff_softmax_handler
# interpret["_softmax"] = lambda x, dim=-1, _half_to_float=False: diff_softmax_handler(x, dim=dim)
interpret["Softmax"] = lambda X, axis=-1: diff_softmax_handler(X, dim=axis)

# Public re-exports
from .relu import relu_linearizer  # noqa: E402, F401
from .tanh import tanh_linearizer  # noqa: E402, F401
from .exp import exp_linearizer  # noqa: E402, F401
from .reciprocal import reciprocal_linearizer  # noqa: E402, F401
from .bilinear import (  # noqa: E402, F401
    diff_bilinear_elementwise,
    diff_bilinear_matmul,
)

__all__ = [
    "interpret",
    "DiffZonoBounds",
    "_register_linearizer",
    "relu_linearizer",
    "tanh_linearizer",
    "exp_linearizer",
    "reciprocal_linearizer",
    "diff_bilinear_elementwise",
    "diff_bilinear_matmul",
    "diff_softmax_handler",
    "diff_heaviside_pruning_handler",
]
