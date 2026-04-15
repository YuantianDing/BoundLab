"""

zmin = min(lx, ly) > 0,  zmax = max(ux, uy)
Smin = −1/zmin²,  Smax = −1/zmax²
δ = max(|l∆|, |u∆|)
λ∆ = (Smin + Smax) / 2,  µ∆ = 0,  β∆ = (Smax − Smin) / 2 · δ

For all lx ≤ x ≤ ux, ly ≤ y ≤ uy, l∆ ≤ x−y ≤ u∆, 0 < lx, 0 < ly:

(Smin+Smax)/2 · (x−y) − (Smax−Smin)/2 · δ  ≤  1/x − 1/y  ≤  (Smin+Smax)/2 · (x−y) + (Smax−Smin)/2 · δ

"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.reciprocal import reciprocal_linearizer as std_reciprocal_linearizer
from . import _register_linearizer, DiffZonoBounds


@_register_linearizer("Reciprocal")
def reciprocal_linearizer(
    xs: list[Expr], ys: list[Expr], ds: list[Expr]
) -> DiffZonoBounds:
    """Return a :class:`DiffZonoBounds` for differential reciprocal.

    Assumes both x and y are strictly positive.
    *x_bounds* and *y_bounds* are standard DeepT reciprocal linearizations.
    *diff_bounds* over-approximates ``1/x − 1/y`` using the
    derivative-range relaxation from the synthesis document.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.diff.zono3.reciprocal import reciprocal_linearizer
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

    # Standard per-branch linearizations
    x_bounds = std_reciprocal_linearizer(x_ub, x_lb)
    y_bounds = std_reciprocal_linearizer(y_ub, y_lb)

    # zmin = min(lx, ly) > 0,  zmax = max(ux, uy)
    z_min = torch.clamp(torch.minimum(x_lb, y_lb), min=1e-9)
    z_max = torch.clamp(torch.maximum(x_ub, y_ub), min=z_min + 1e-12)

    # Smin = −1/zmin²,  Smax = −1/zmax²
    # Note: since zmin < zmax, |Smin| > |Smax|, so Smin < Smax < 0
    s_min = -1.0 / (z_min ** 2)
    s_max = -1.0 / (z_max ** 2)

    # δ = max(|l∆|, |u∆|)
    delta = torch.maximum(d_lb.abs(), d_ub.abs())

    # λ∆ = (Smin + Smax) / 2,  µ∆ = 0,  β∆ = (Smax − Smin) / 2 · δ
    sd = (s_min + s_max) / 2.0
    bias = torch.zeros_like(x_ub)
    err = (s_max - s_min) / 2.0 * delta  # Smax > Smin so this is positive

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