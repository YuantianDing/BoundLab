"""Exp linearizer for polytope abstract interpretation."""

import torch

from . import PolyBounds, _register_linearizer


@_register_linearizer("Exp")
def exp_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> PolyBounds:
    r"""CROWN relaxation of :math:`\exp`.

    Uses the tight secant line as the upper envelope and a tangent at
    the bound midpoint as the lower envelope, both convex-guaranteed
    on :math:`[\ell, u]`.

    Degenerate intervals (:math:`u \approx \ell`) collapse to the exact
    tangent at :math:`\ell`.
    """
    degen = torch.abs(ub - lb) < 1e-12

    exp_lb = torch.exp(lb)
    exp_ub = torch.exp(ub)

    # Upper envelope: secant (tight for convex exp)
    denom = (ub - lb).clamp(min=1e-30)
    upper_lam = (exp_ub - exp_lb) / denom
    upper_bias = exp_lb - upper_lam * lb

    # Lower envelope: tangent at the midpoint
    mid = 0.5 * (ub + lb)
    exp_mid = torch.exp(mid)
    lower_lam = exp_mid
    lower_bias = exp_mid * (1.0 - mid)

    # Degenerate case: the exact tangent at lb.
    exp_tangent = torch.exp(lb)
    upper_lam = torch.where(degen, exp_tangent, upper_lam)
    upper_bias = torch.where(degen, exp_tangent * (1.0 - lb), upper_bias)
    lower_lam = torch.where(degen, exp_tangent, lower_lam)
    lower_bias = torch.where(degen, exp_tangent * (1.0 - lb), lower_bias)

    return PolyBounds(
        upper_lam=upper_lam,
        upper_bias=upper_bias,
        lower_lam=lower_lam,
        lower_bias=lower_bias,
    )
