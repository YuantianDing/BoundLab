"""Differential exp linearizer — hexagon-Chebyshev.

Output form (unchanged from paper):
    Ẑ_Δ = λ_Δ · Z_Δ + μ_Δ + β_Δ · ε_new,   μ_Δ = 0

Change: (λ_Δ, β_Δ) computed from the range of the slope function
    S(x, y) = (e^x - e^y) / (x - y)
over the feasible hexagon P = [lx, ux]×[ly, uy] ∩ {lΔ ≤ x−y ≤ uΔ}, rather
than from the range of f' = exp over the merged interval [L, U].

Since S(P) ⊆ exp([L, U]) with typically strict inclusion, this produces a
β_Δ never larger than the paper's — and dramatically smaller when the x- and
y-boxes separate.

Soundness: for all (x, y) ∈ P,
    λ_Δ (x - y) - β_Δ  ≤  e^x - e^y  ≤  λ_Δ (x - y) + β_Δ.
"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.exp import exp_linearizer as std_exp_linearizer
from . import _register_linearizer, DiffZonoBounds
from ._hex_cheby import hex_chebyshev_transfer, slope_exp


@_register_linearizer("Exp")
def exp_linearizer(
    xs: list[Expr], ys: list[Expr], ds: list[Expr]
) -> DiffZonoBounds:
    """Differential exp linearizer using hexagon-Chebyshev β.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.diff.zono3.exp import exp_linearizer
    >>> x = expr.ConstVal(torch.tensor([1.0])) + 0.2 * expr.LpEpsilon([1])
    >>> y = expr.ConstVal(torch.tensor([0.5])) + 0.2 * expr.LpEpsilon([1])
    >>> d = x - y
    >>> dzb = exp_linearizer([x], [y], [d])
    >>> dzb.diff_bounds.bias.shape
    torch.Size([1])
    """
    x, y, diff = xs[0], ys[0], ds[0]

    x_ub, x_lb = ublb(x)
    y_ub, y_lb = ublb(y)
    d_ub, d_lb = ublb(diff)
    ndim = len(x_ub.shape)

    x_bounds = std_exp_linearizer(x_ub, x_lb)
    y_bounds = std_exp_linearizer(y_ub, y_lb)

    lambda_d, mu_d, beta_d = hex_chebyshev_transfer(
        slope_exp, x_lb, x_ub, y_lb, y_ub, d_lb, d_ub,
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
