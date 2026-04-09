"""

Smin = exp(min(lx, ly)),  Smax = exp(max(ux, uy))
δ = max(|l∆|, |u∆|)
λ∆ = (Smin + Smax) / 2,  µ∆ = 0,  β∆ = (Smax − Smin) / 2 · δ

For all lx ≤ x ≤ ux, ly ≤ y ≤ uy, l∆ ≤ x−y ≤ u∆:

(Smin+Smax)/2 · (x−y) − (Smax−Smin)/2 · δ  ≤  exp(x)−exp(y)  ≤  (Smin+Smax)/2 · (x−y) + (Smax−Smin)/2 · δ

"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.exp import exp_linearizer as std_exp_linearizer
from . import _register_linearizer, DiffZonoBounds


@_register_linearizer("exp")
def exp_linearizer(
    xs: list[Expr], ys: list[Expr], ds: list[Expr]
) -> DiffZonoBounds:
    """Return a :class:`DiffZonoBounds` for differential exp.

    *x_bounds* and *y_bounds* are standard DeepT exp linearizations.
    *diff_bounds* over-approximates ``exp(x) − exp(y)`` using the
    mean-derivative relaxation from the synthesis document.

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

    # Standard per-branch linearizations
    x_bounds = std_exp_linearizer(x_ub, x_lb)
    y_bounds = std_exp_linearizer(y_ub, y_lb)

    # Smin = exp(min(lx, ly)),  Smax = exp(max(ux, uy))
    z_lo = torch.minimum(x_lb, y_lb)
    z_hi = torch.maximum(x_ub, y_ub)
    z_lo_c = torch.clamp(z_lo, -30, 30)
    z_hi_c = torch.clamp(z_hi, -30, 30)
    s_min = torch.exp(z_lo_c)
    s_max = torch.exp(z_hi_c)

    # δ = max(|l∆|, |u∆|)
    delta = torch.maximum(d_lb.abs(), d_ub.abs())

    # λ∆ = (Smin + Smax) / 2,  µ∆ = 0,  β∆ = (Smax − Smin) / 2 · δ
    sd = (s_min + s_max) / 2.0
    bias = torch.zeros_like(x_ub)
    err = (s_max - s_min) / 2.0 * delta

    # degenerate case
    degen = delta < 1e-15
    sd = torch.where(degen, s_min, sd)
    err = torch.where(degen, torch.zeros_like(err), err)

    return DiffZonoBounds(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        diff_bounds=ZonoBounds(
            bias=bias,
            error_coeffs=EinsumOp.from_hardmard(err, ndim),
            input_weights=[sd],
        ),
        diff_x_error=0,
        diff_x_weights=0,
        diff_y_error=0,
        diff_y_weights=0,
    )