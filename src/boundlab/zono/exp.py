"""Exp linearizer for zonotope abstract interpretation.

Implements the DeepT minimal-area relaxation for the exponential function.
"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp

from . import ZonoBounds, _register_linearizer


@_register_linearizer("exp")
def exp_linearizer(expr: Expr) -> ZonoBounds:
    """Minimal-area exp relaxation (DeepT, Section 4.5).

    For each element with input bounds [l, u]:

    - Degenerate (u ≈ l): output is exp(l), no error.
    - General: tangent line at optimal point t_opt as lower bound,
      secant between (l, exp(l)) and (u, exp(u)) as upper bound.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.exp import exp_linearizer
    >>> x = expr.ConstVal(torch.tensor([0.0])) + 0.1 * expr.LpEpsilon([1])
    >>> b = exp_linearizer(x)
    >>> b.bias.shape
    torch.Size([1])
    """
    lb = expr.lb()
    ub = expr.ub()
    output_shape = ub.shape

    lb_c = torch.clamp(lb, -30, 30)
    ub_c = torch.clamp(ub, -30, 30)
    el = torch.exp(lb_c)
    eu = torch.exp(ub_c)

    degen = torch.abs(ub - lb) < 1e-12

    # Secant slope
    secant_slope = torch.where(degen, el, (eu - el) / (ub - lb + 1e-30))

    # Optimal tangent point (minimal area)
    t_crit = torch.log(torch.clamp(secant_slope, min=1e-30))
    t_opt = torch.minimum(t_crit, lb + 1.0 - 0.01)

    # Slope = exp(tangent_point)
    slope = torch.exp(torch.clamp(t_opt, -30, 30))

    # Bias and error
    et = slope  # exp(t_opt)
    mu = 0.5 * (et - slope * t_opt + eu - slope * ub)
    beta = 0.5 * (slope * t_opt - et + eu - slope * ub)

    slope = torch.where(degen, torch.zeros_like(slope), slope)
    mu = torch.where(degen, el, mu)
    beta = torch.where(degen, torch.zeros_like(beta), torch.abs(beta))

    # Build ZonoBounds
    
    error_op = EinsumOp.from_hardmard(beta, len(expr.shape))

    return ZonoBounds(bias=mu, error_coeffs=error_op, input_weights=[slope])
