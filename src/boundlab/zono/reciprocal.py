"""Reciprocal linearizer for zonotope abstract interpretation.

Implements the DeepT minimal-area relaxation for the reciprocal function (1/x).
"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp

from . import ZonoBounds, _register_linearizer


@_register_linearizer("reciprocal")
def reciprocal_linearizer(expr: Expr) -> ZonoBounds:
    """Minimal-area reciprocal relaxation (DeepT, Section 4.6).

    Assumes input is strictly positive. Clamps lower bound to 1e-9.
    """
    lb = expr.lb()
    ub = expr.ub()
    output_shape = ub.shape

    # Clamp to positive
    lb = torch.clamp(lb, min=1e-9)
    ub = torch.clamp(ub, min=lb + 1e-12)

    degen = torch.abs(ub - lb) < 1e-12

    # Optimal tangent point (geometric mean minimizes relaxation area)
    t_opt = torch.sqrt(ub * lb)

    slope = -1.0 / (t_opt ** 2)
    val_at_t = 1.0 / t_opt

    # Lower bound line (tangent at t_opt): intercept = 2/t
    c_lower = val_at_t - slope * t_opt  # = 2/t
    # Upper bound line: intercept = max over endpoints of (1/x - slope*x)
    c_upper = torch.maximum(1.0 / lb - slope * lb, 1.0 / ub - slope * ub)

    mu = 0.5 * (c_upper + c_lower)
    beta = 0.5 * (c_upper - c_lower)

    slope = torch.where(degen, torch.zeros_like(slope), slope)
    mu = torch.where(degen, 1.0 / lb, mu)
    beta = torch.where(degen, torch.zeros_like(beta), torch.abs(beta))
    
    error_op = EinsumOp.from_hardmard(beta, len(expr.shape))
    return ZonoBounds(bias=mu, error_coeffs=error_op, input_weights=[slope])