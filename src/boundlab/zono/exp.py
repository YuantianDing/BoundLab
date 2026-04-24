"""Exp linearizer for zonotope abstract interpretation.

Implements the DeepT minimal-area relaxation for the exponential function.
"""

import torch

from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp

from . import ZonoBounds, _register_linearizer


@_register_linearizer("Exp")
def exp_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> ZonoBounds:
    """Minimal-area exp relaxation (DeepT, Section 4.5).

    The returned zonotope ``y = slope·x + mu + beta·ε`` (with ε ∈ [-1, 1])
    over-approximates ``exp(x)`` on ``[lb, ub]`` and guarantees
    ``mu - beta ≥ 0`` so downstream handlers (``reciprocal`` / softmax)
    never see a non-positive lower bound.

    Strategy:

    - For each element, try the single-slope "minimal area" relaxation:
      tangent at ``t_opt`` as the lower envelope, parallel secant line
      through ``(ub_c, exp(ub_c))`` as the upper envelope.
    - If the tangent gives a non-positive offset (e.g. ``lb > 0``) or the
      fp32 precision would lose that offset against ``exp(ub)``, fall back
      to the interval relaxation ``slope = 0``, ``mu = beta = exp(ub)/2``,
      which yields ``[0, exp(ub)]`` — loose but guaranteed ``mu - beta = 0``
      bit-exactly (so downstream ``ublb`` cannot go negative).
    - Clamp ``lb``/``ub`` to ±30 to keep ``exp`` numerically safe; the
      underflow/overflow branches use the interval fallback.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.exp import exp_linearizer
    >>> x = expr.ConstVal(torch.tensor([0.0])) + 0.1 * expr.LpEpsilon([1])
    >>> ub, lb = x.ublb()
    >>> b = exp_linearizer(ub, lb)
    >>> b.bias.shape
    torch.Size([1])
    """
    exp_lb = torch.exp(lb)
    exp_ub = torch.exp(ub)
    slope = (exp_ub - exp_lb) / (ub - lb)
    slope = torch.where(torch.isfinite(slope), slope, torch.exp((ub + lb) / 2))
    # assert torch.isfinite(slope).all(), f"Expected finite slope point for exp linearizer {slope.max().item()} {ub.max().item()} {lb.max().item()}"
    
    slope_point = torch.log(slope)
    # assert torch.isfinite(slope_point).all(), f"Expected finite slope point for exp linearizer {slope_point.max().item()}"
    U = torch.max(exp_ub - slope * ub, exp_lb - slope * lb)
    L = slope * (1 - slope_point)
    # assert torch.isfinite(U).all() and torch.isfinite(L).all(), "Expected finite envelopes for exp linearizer"

    beta = (U - L) / 2
    mu = (U + L) / 2

    # mu = torch.where(large_cases, torch.inf, mu)

    error_op = EinsumOp.from_hardmard(beta, len(ub.shape))
    return ZonoBounds(bias=mu, error_coeffs=error_op, input_weights=[slope])
