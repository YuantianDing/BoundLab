"""Tanh linearizer for polytope abstract interpretation."""

import torch

from . import PolyBounds, _register_linearizer


@_register_linearizer("tanh")
def tanh_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> PolyBounds:
    r"""CROWN relaxation of :math:`\tanh`.

    ``tanh`` is concave for :math:`x \ge 0` and convex for :math:`x \le 0`.
    For each neuron we use the common CROWN envelope:

    - When both bounds share a sign, the tight secant forms the concave-
      side envelope and a tangent (at the bound midpoint) forms the
      convex-side envelope.
    - For sign-crossing intervals we use the minimum-slope line through
      the respective endpoint on each side, giving a sound (though not
      tightest-possible) relaxation.
    """
    degen = torch.abs(ub - lb) < 1e-12

    tl = torch.tanh(lb)
    tu = torch.tanh(ub)

    denom = (ub - lb).clamp(min=1e-30)
    secant = (tu - tl) / denom
    secant_bias_u = tu - secant * ub  # tu − slope·ub == tl − slope·lb

    # Tangent at the midpoint (slope 1 − tanh²(m)).
    mid = 0.5 * (lb + ub)
    tm = torch.tanh(mid)
    tangent_slope = 1.0 - tm * tm
    tangent_bias = tm - tangent_slope * mid

    safe_slope = torch.minimum(1.0 - tl * tl, 1.0 - tu * tu)

    non_negative = lb >= 0
    non_positive = ub <= 0

    upper_lam = torch.where(
        non_positive,
        secant,
        torch.where(non_negative, tangent_slope, safe_slope),
    )
    upper_bias = torch.where(
        non_positive,
        secant_bias_u,
        torch.where(non_negative, tangent_bias, tu - safe_slope * ub),
    )
    lower_lam = torch.where(
        non_negative,
        secant,
        torch.where(non_positive, tangent_slope, safe_slope),
    )
    lower_bias = torch.where(
        non_negative,
        secant_bias_u,
        torch.where(non_positive, tangent_bias, tl - safe_slope * lb),
    )

    exact_lam = 1.0 - tl * tl
    exact_bias = tl - exact_lam * lb
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
