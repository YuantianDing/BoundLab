"""

σ = min(sech²(min(lx, ly)), sech²(max(ux, uy)))
δ = max(|l∆|, |u∆|)
λ∆ = (1+σ)/2,  µ∆ = 0,  β∆ = (1−σ)/2 · δ

For all lx ≤ x ≤ ux, ly ≤ y ≤ uy, l∆ ≤ x−y ≤ u∆:

(1+σ)/2 · (x−y) − (1−σ)/2 · δ  ≤  tanh(x) − tanh(y)  ≤  (1+σ)/2 · (x−y) + (1−σ)/2 · δ

"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.tanh import tanh_linearizer as std_tanh_linearizer
from .. import DiffZonoBounds


def tanh_linearizer(
    xs: list[Expr], ys: list[Expr], ds: list[Expr]
) -> DiffZonoBounds:
    """Return a :class:`DiffZonoBounds` for differential tanh.

    *x_bounds* and *y_bounds* are standard DeepT tanh linearizations.
    *diff_bounds* over-approximates ``tanh(x) − tanh(y)`` using the
    minimum-slope relaxation from the synthesis document.

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

    # Standard per-branch linearizations
    x_bounds = std_tanh_linearizer(x_ub, x_lb)
    y_bounds = std_tanh_linearizer(y_ub, y_lb)

    # σ = min(sech²(min(lx, ly)), sech²(max(ux, uy)))
    z_lo = torch.minimum(x_lb, y_lb)
    z_hi = torch.maximum(x_ub, y_ub)
    sech2_lo = 1.0 - torch.tanh(z_lo) ** 2
    sech2_hi = 1.0 - torch.tanh(z_hi) ** 2
    sigma = torch.minimum(sech2_lo, sech2_hi)

    # δ = max(|l∆|, |u∆|)
    delta = torch.maximum(d_lb.abs(), d_ub.abs())

    # λ∆ = (1+σ)/2,  µ∆ = 0,  β∆ = (1−σ)/2 · δ
    sd = (1.0 + sigma) / 2.0
    bias = torch.zeros_like(x_ub)
    err = (1.0 - sigma) / 2.0 * delta

    # degenerate case
    degen = delta < 1e-15
    sd = torch.where(degen, torch.ones_like(sd), sd)
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