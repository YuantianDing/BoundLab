"""Reciprocal linearizer for zonotope abstract interpretation.

Implements the DeepT minimal-area relaxation for the reciprocal function (1/x).
"""

import torch

from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp

from . import ZonoBounds, _register_linearizer


@_register_linearizer("Reciprocal")
def reciprocal_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> ZonoBounds:
    """Minimal-area reciprocal relaxation with positive-output constraint.

    Matches DeepT (Bonaert et al., 2021, Section 4.6) with the
    ``y_positive_constraint`` that prevents bound explosion when the
    input lower bound is near zero (as happens inside softmax when
    ``sum(exp(x_j - x_i))`` has a loose lower bound).

    The key change from the naive relaxation: the tangent point is
    clamped to ``t_opt >= ub/2 + 0.01``, ensuring the reciprocal
    output remains strictly positive and the relaxation stays tight.
    """
    output_shape = ub.shape

    # Clamp to positive
    # lb = torch.clamp(lb, min=1e-9)
    # ub = torch.clamp(ub, min=lb + 1e-12)

    degen = torch.abs(ub - lb) < 1e-12

    # Optimal tangent point: geometric mean minimizes area, but
    # clamp to ub/2 + 0.01 to ensure strictly positive output
    t_crit = torch.sqrt(ub * lb)
    t_crit2 = 0.5 * ub + 0.01
    t_opt = torch.maximum(t_crit, t_crit2)

    slope = -1.0 / (t_opt ** 2)
    val_at_t = 1.0 / t_opt

    # Lower bound line (tangent at t_opt)
    c_lower = val_at_t - slope * t_opt  # = 2/t_opt
    # Upper bound line: connects through whichever endpoint gives
    # the tighter (lower) intercept
    c_upper = torch.maximum(1.0 / lb - slope * lb, 1.0 / ub - slope * ub)

    mu = 0.5 * (c_upper + c_lower)
    beta = 0.5 * (c_upper - c_lower)

    slope = torch.where(degen, torch.zeros_like(slope), slope)
    mu = torch.where(degen, 1.0 / lb, mu)
    beta = torch.where(degen, torch.zeros_like(beta), torch.abs(beta))
    
    error_op = EinsumOp.from_hardmard(beta, len(ub.shape))
    return ZonoBounds(bias=mu, error_coeffs=error_op, input_weights=[slope])
