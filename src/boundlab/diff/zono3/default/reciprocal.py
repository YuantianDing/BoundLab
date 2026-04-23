"""Differential reciprocal linearizer — hexagon-Chebyshev.

Output form (unchanged from paper):
    Ẑ_Δ = λ_Δ · Z_Δ + μ_Δ + β_Δ · ε_new,   μ_Δ = 0

Change: (λ_Δ, β_Δ) computed from the range of the slope function
    S(x, y) = -1 / (x y)
over the feasible hexagon P, rather than from [-1/z_min², -1/z_max²] where
z_min = min(lx, ly), z_max = max(ux, uy). Since {xy : (x,y) ∈ P} is pinned
by the corners (x y monotone ↑ in both), we get S(P) ⊆ [-1/(z_min²), -1/(z_max²)]
strictly.

Soundness: for all (x, y) ∈ P, 0 < lx, 0 < ly:
    λ_Δ (x - y) - β_Δ  ≤  1/x - 1/y  ≤  λ_Δ (x - y) + β_Δ.
"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.reciprocal import reciprocal_linearizer as std_reciprocal_linearizer
from .. import DiffZonoBounds
from ._hex_cheby import hex_chebyshev_transfer, slope_recip


def reciprocal_linearizer(
    xs: list[Expr], ys: list[Expr], ds: list[Expr]
) -> DiffZonoBounds:
    """Differential 1/x linearizer using hexagon-Chebyshev β.

    Assumes both x and y are strictly positive.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.diff.zono3.default.reciprocal import reciprocal_linearizer
    >>> x = expr.ConstVal(torch.tensor([2.0])) + 0.1 * expr.LpEpsilon([1])
    >>> y = expr.ConstVal(torch.tensor([3.0])) + 0.1 * expr.LpEpsilon([1])
    >>> d = x - y
    >>> dzb = reciprocal_linearizer([x], [y], [d])
    >>> dzb.diff_bounds.bias.shape
    torch.Size([1])
    """
    x, y, diff = xs[0], ys[0], ds[0]

    x_ub, x_lb = ublb(x)
    y_ub, y_lb = ublb(y)
    d_ub, d_lb = ublb(diff)
    ndim = len(x_ub.shape)

    # Clamp to strictly positive for safety.
    x_lb = torch.clamp(x_lb, min=1e-9)
    x_ub = torch.clamp(x_ub, min=x_lb + 1e-12)
    y_lb = torch.clamp(y_lb, min=1e-9)
    y_ub = torch.clamp(y_ub, min=y_lb + 1e-12)

    x_bounds = std_reciprocal_linearizer(x_ub, x_lb)
    y_bounds = std_reciprocal_linearizer(y_ub, y_lb)

    lambda_d, mu_d, beta_d = hex_chebyshev_transfer(
        slope_recip, x_lb, x_ub, y_lb, y_ub, d_lb, d_ub,
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
