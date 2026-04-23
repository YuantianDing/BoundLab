"""Differential tanh linearizer — hexagon-Chebyshev.

Output form (unchanged from paper):
    Ẑ_Δ = λ_Δ · Z_Δ + μ_Δ + β_Δ · ε_new,   μ_Δ = 0

Change: (λ_Δ, β_Δ) computed from the range of the slope function
    S(x, y) = (tanh x - tanh y) / (x - y)
over the feasible hexagon P, rather than from [σ, 1] where
σ = min(sech²(L), sech²(U)). The paper's rule uses 1 as an upper bound on
sech²; the hexagon-Chebyshev form uses the *actual* max slope, which is
≤ 1 always and ≪ 1 in the saturation regime (|x|, |y| large same sign or
opposite signs far from 0). This is the case that blows up most dramatically
for the paper — β shrinks by factors up to ~10^10 for deeply-saturated inputs.

Soundness: for all (x, y) ∈ P,
    λ_Δ (x - y) - β_Δ  ≤  tanh(x) - tanh(y)  ≤  λ_Δ (x - y) + β_Δ.
"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.tanh import tanh_linearizer as std_tanh_linearizer
from .. import DiffZonoBounds
from ._hex_cheby import (
    hex_chebyshev_transfer, slope_tanh, tanh_extra_candidates,
    tanh_edge_critical_bounds,
)


def tanh_linearizer(
    xs: list[Expr], ys: list[Expr], ds: list[Expr]
) -> DiffZonoBounds:
    """Differential tanh linearizer using hexagon-Chebyshev β.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.diff.zono3.default.tanh import tanh_linearizer
    >>> x = expr.ConstVal(torch.tensor([0.5])) + 0.3 * expr.LpEpsilon([1])
    >>> y = expr.ConstVal(torch.tensor([0.2])) + 0.3 * expr.LpEpsilon([1])
    >>> d = x - y
    >>> dzb = tanh_linearizer([x], [y], [d])
    >>> dzb.diff_bounds.bias.shape
    torch.Size([1])
    """
    x, y, diff = xs[0], ys[0], ds[0]

    x_ub, x_lb = ublb(x)
    y_ub, y_lb = ublb(y)
    d_ub, d_lb = ublb(diff)
    ndim = len(x_ub.shape)

    x_bounds = std_tanh_linearizer(x_ub, x_lb)
    y_bounds = std_tanh_linearizer(y_ub, y_lb)

    lambda_d, mu_d, beta_d = hex_chebyshev_transfer(
        slope_tanh, x_lb, x_ub, y_lb, y_ub, d_lb, d_ub,
        extra_candidates=tanh_extra_candidates(x_lb, x_ub, y_lb, y_ub),
        extra_smax=tanh_edge_critical_bounds(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub),
    )

    degen = torch.maximum(d_lb.abs(), d_ub.abs()) < 1e-15
    beta_d = torch.where(degen, torch.zeros_like(beta_d), beta_d)

    return DiffZonoBounds(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        diff_bounds=ZonoBounds(
            bias=mu_d,
            error_coeffs=EinsumOp.from_hardmard(beta_d, ndim),
            input_weights=[lambda_d],
        ),
        diff_x_error=0,
        diff_x_weights=0,
        diff_y_error=0,
        diff_y_weights=0,
    )
