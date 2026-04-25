"""Reciprocal linearizer for polytope abstract interpretation."""

import torch

from . import PolyBounds, _register_linearizer


@_register_linearizer("Reciprocal")
def reciprocal_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> PolyBounds:
    r"""CROWN relaxation of :math:`1/x` on a strictly positive domain.

    ``1/x`` is convex on :math:`(0, \infty)`:

    - Upper envelope: secant through :math:`(\ell, 1/\ell)` and
      :math:`(u, 1/u)`.
    - Lower envelope: tangent at a point :math:`t_{\mathrm{opt}}`. The
      minimum-area tangent point is :math:`\sqrt{\ell u}`; we additionally
      clamp it to :math:`u/2 + 0.01` to keep the lower envelope strictly
      positive (important when feeding softmax denominators).
    """
    degen = torch.abs(ub - lb) < 1e-12

    denom = (ub - lb).clamp(min=1e-30)
    upper_lam = (1.0 / ub - 1.0 / lb) / denom
    upper_bias = 1.0 / lb - upper_lam * lb

    t_crit = torch.sqrt(ub.clamp(min=1e-30) * lb.clamp(min=1e-30))
    t_opt = torch.maximum(t_crit, 0.5 * ub + 0.01)
    lower_lam = -1.0 / (t_opt * t_opt)
    lower_bias = 2.0 / t_opt  # 1/t − lower_lam · t == 1/t + 1/t

    exact_lam = -1.0 / (lb * lb)
    exact_bias = 1.0 / lb - exact_lam * lb
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
