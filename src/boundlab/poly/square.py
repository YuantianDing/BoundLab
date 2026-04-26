"""Square linearizer for polytope abstract interpretation."""

from __future__ import annotations

import torch

from . import PolyBounds, _register_linearizer


@_register_linearizer("Square")
def square_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> PolyBounds:
    r"""CROWN relaxation of :math:`x^2`.

    Uses the secant line as the upper envelope and the tangent at the
    interval midpoint as the lower envelope.
    """
    degen = torch.abs(ub - lb) < 1e-12

    # Upper envelope (secant through (lb, lb^2) and (ub, ub^2)).
    upper_lam = ub + lb
    upper_bias = -ub * lb

    # Lower envelope (tangent at midpoint).
    mid = 0.5 * (ub + lb)
    lower_lam = 2.0 * mid
    lower_bias = -(mid * mid)

    # Degenerate interval: exact line at x = lb.
    exact_lam = 2.0 * lb
    exact_bias = -(lb * lb)
    upper_lam = torch.where(degen, exact_lam, upper_lam)
    upper_bias = torch.where(degen, exact_bias, upper_bias)
    lower_lam = torch.where(degen, exact_lam, lower_lam)
    lower_bias = torch.where(degen, exact_bias, lower_bias)

    return PolyBounds(
        upper_lam=upper_lam,
        upper_bias=upper_bias,
        lower_lam=lower_lam,
        lower_bias=lower_bias,
    )
