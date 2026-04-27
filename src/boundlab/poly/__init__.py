r"""Polytope-Based Abstract Interpretation for Neural Networks.

This module provides CROWN-style abstract interpretation using linear
polytope relaxations of nonlinear activations. Each neuron is bounded
by a pair of linear envelopes:

.. math::

    \lambda_\ell \odot x + b_\ell \;\le\; f(x) \;\le\; \lambda_u \odot x + b_u

The central :class:`PolyBoundGate` expression represents an abstract
function with fixed :math:`\pm 1` offsets; general CROWN-style bounds
are wrapped by rescaling around their midpoint.

Examples
--------
>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> import boundlab.poly as poly
>>> model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
>>> op = poly.interpret(model)
>>> x = expr.ConstVal(torch.zeros(4)) + expr.LpEpsilon([4])
>>> y = op(x)
>>> y.ub().shape
torch.Size([3])
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Literal

import torch

from boundlab import expr as _expr
from boundlab.expr._affine import AffineSum, ConstVal
from boundlab.expr._core import Expr, ExprFlags
from boundlab.interp import ONNX_BASE_INTERPRETER, Interpreter
from boundlab.linearop import LinearOp
from boundlab.linearop._einsum import EinsumOp


interpret = Interpreter[Expr](ONNX_BASE_INTERPRETER)
"""Polytope-based interpreter.

Dispatches neural-network operators to CROWN-style linearizers that
produce :class:`PolyBoundGate`-wrapped expressions.

Examples
--------
>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> import boundlab.poly as poly
>>> op = poly.interpret(nn.Linear(2, 1))
>>> y = op(expr.ConstVal(torch.zeros(2)) + expr.LpEpsilon([2]))
>>> y.shape
torch.Size([1])
"""


class PolyBoundGate(Expr):
    r"""Abstract gate bounded pointwise by a pair of linear polytopes on its child.

    Represents an elementwise function :math:`f(x)` whose output is constrained by

    .. math::
        \lambda_\ell \odot x - 1 \;\le\; f(x) \;\le\; \lambda_u \odot x + 1,

    where :math:`x` is the child expression and :math:`\lambda_u, \lambda_\ell`
    (``upper_lam``, ``lower_lam``) are concrete tensors of the same shape
    as the child.

    Backward propagation splits the incoming weight :math:`w` by sign
    element-wise on its materialized Jacobian. For direction ``"<="``:

    .. math::
        w \cdot f(x) \;\le\; (w_+ \odot \lambda_u + w_- \odot \lambda_\ell)\,x
            + \sum_j |w_{\cdot j}|,

    and symmetrically for ``">="`` with the slopes swapped and the bias negated.
    """

    def __init__(self, child: Expr, upper_lam: torch.Tensor, lower_lam: torch.Tensor,
                 *, reason: str | None = None):
        super().__init__(ExprFlags.NONE)
        assert child.shape == upper_lam.shape == lower_lam.shape, (
            f"Shape mismatch: child={child.shape}, "
            f"upper_lam={upper_lam.shape}, lower_lam={lower_lam.shape}"
        )
        self._child = child
        self.upper_lam = upper_lam
        self.lower_lam = lower_lam
        self.reason = reason if reason is not None else str(inspect.stack()[1].function)

    @property
    def shape(self) -> torch.Size:
        return self._child.shape

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self._child,)

    def with_children(self, *new_children: Expr) -> "PolyBoundGate":
        (new_child,) = new_children
        return PolyBoundGate(new_child, self.upper_lam, self.lower_lam, reason=self.reason)

    def backward(self, weights: LinearOp, direction: Literal[">=", "<=", "=="]):
        if direction == "==":
            return None

        jac = weights.jacobian()
        in_ndim = len(weights.input_shape)

        jac_pos = jac.clamp(min=0)
        jac_neg = jac.clamp(max=0)

        if direction == "<=":
            new_jac = jac_pos * self.upper_lam + jac_neg * self.lower_lam
            bias_sign = 1.0
        else:
            new_jac = jac_pos * self.lower_lam + jac_neg * self.upper_lam
            bias_sign = -1.0

        abs_jac = jac.abs()
        if in_ndim > 0:
            reduce_dims = tuple(range(jac.dim() - in_ndim, jac.dim()))
            bias = abs_jac.sum(dim=reduce_dims)
        else:
            bias = abs_jac
        bias = bias_sign * bias

        new_op = EinsumOp.from_full(new_jac, in_ndim)
        return bias, [new_op]

    def to_string(self, child_str: str) -> str:
        return f"PolyBoundGate({child_str})"


@dataclasses.dataclass
class PolyBounds:
    r"""CROWN-style linear relaxation bounds for a unary nonlinearity.

    Represents pointwise constraints

    .. math::
        \lambda_\ell \odot x + b_\ell \;\le\; f(x) \;\le\; \lambda_u \odot x + b_u.

    The constituent tensors are per-neuron and share the activation's
    input/output shape.
    """

    upper_lam: torch.Tensor
    upper_bias: torch.Tensor
    lower_lam: torch.Tensor
    lower_bias: torch.Tensor


def _bounds_to_expr(x: Expr, bounds: "PolyBounds", *, eps: float = 1e-30,
                    reason: str | None = None) -> Expr:
    r"""Wrap ``x`` as an expression satisfying ``bounds``.

    Expresses the CROWN relaxation as

    .. math::
        f(x) \;=\; \bar{\lambda} \odot x + \bar{b}
            + \beta \odot \mathrm{PolyBoundGate}(x, U, L),

    where :math:`\bar{\lambda}, \bar{b}` are the midpoint slopes/biases,
    :math:`\beta = (b_u - b_\ell)/2` is the bias half-width, and
    :math:`U = -L = (\lambda_u - \lambda_\ell)/(2\beta)`. Neurons with a
    tight bound (:math:`\beta = 0`) contribute only the affine part.
    """
    ul, ub = bounds.upper_lam, bounds.upper_bias
    ll, lb = bounds.lower_lam, bounds.lower_bias

    base_lam = 0.5 * (ul + ll)
    base_bias = 0.5 * (ub + lb)
    err = 0.5 * (ub - lb)
    slope_slack = 0.5 * (ul - ll)

    exact = err <= eps
    err_safe = torch.where(exact, torch.ones_like(err), err)
    U = torch.where(exact, torch.zeros_like(ul), slope_slack / err_safe)
    L = -U
    err = torch.where(exact, torch.zeros_like(err), err)

    gate = PolyBoundGate(x, U, L, reason=reason)
    return base_lam * x + base_bias + err * gate


def _register_linearizer(name: str):
    r"""Register a CROWN-style linearizer under ``name`` in :data:`interpret`.

    The decorated function takes pairs of concrete ``(ub, lb)`` tensors
    — one pair per input expression — and returns a :class:`PolyBounds`.
    The registered handler evaluates bounds via :func:`~boundlab.prop.ublb`,
    invokes the linearizer, and wraps the result with
    :class:`PolyBoundGate` via :func:`_bounds_to_expr`.
    """

    def decorator(linearizer):
        def handler(*exprs: Expr) -> Expr:
            if all(isinstance(e, ConstVal) for e in exprs):
                return NotImplemented
            assert len(exprs) == 1, \
                "Only unary linearizers are supported; got {} inputs.".format(len(exprs))
            (x,) = exprs
            ub, lb = x.ublb()
            bounds = linearizer(ub, lb)
            assert (
                bounds.upper_lam.shape == x.shape
                and bounds.lower_lam.shape == x.shape
                and bounds.upper_bias.shape == x.shape
                and bounds.lower_bias.shape == x.shape
            ), "PolyBounds tensors must match the input expression shape."
            return _bounds_to_expr(x, bounds, reason=linearizer.__name__)

        interpret[name] = handler
        return linearizer

    return decorator


# =====================================================================
# Import activation modules — each calls _register_linearizer
# =====================================================================

from .relu import relu_linearizer
from .exp import exp_linearizer
from .reciprocal import reciprocal_linearizer
from .tanh import tanh_linearizer
from .square import square_linearizer

# ONNX activation handlers
interpret["Relu"] = interpret["relu"]
interpret["Tanh"] = interpret["tanh"]

# Softmax
from .softmax import softmax_handler
from .softmax2 import softmax2_handler
from .bilinear import matmul_handler, mul_handler
interpret["Softmax"] = lambda X, axis=-1: softmax_handler(X, dim=axis)
interpret["Softmax2"] = softmax2_handler
interpret["MatMul"] = matmul_handler
interpret["Mul"] = mul_handler

__all__ = [
    "PolyBoundGate",
    "PolyBounds",
    "interpret",
    "relu_linearizer",
    "exp_linearizer",
    "reciprocal_linearizer",
    "tanh_linearizer",
    "square_linearizer",
    "softmax_handler",
    "softmax2_handler",
    "matmul_handler",
    "mul_handler",
]
