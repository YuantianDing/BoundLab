"""Differential ReLU lineariser (VeryDiff Table 1, 9-case split).

Cases 1–8 reduce to ``relu(x) − relu(y)`` via the standard triangle
relaxation applied independently to *x* and *y*.  The error epsilons
introduced for *x* and *y* are **shared** with the diff component, so
the diff expression tracks ``x_output − y_output`` exactly for those
cases (no extra approximation error).

Case 9 (both crossing) uses a tighter triangle relaxation directly on
``d = x − y`` with slope ``alpha = clamp(d_ub / (d_ub − d_lb), 0, 1)``
and half-width ``mu_d = 0.5 · max(d_ub, −d_lb)``.  A fresh epsilon is
introduced only for case-9 neurons.
"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from . import _register_linearizer, DiffZonoBounds


@_register_linearizer("relu")
def relu_linearizer(
    xs: list[Expr], ys: list[Expr], ds: list[Expr]
) -> DiffZonoBounds:
    """Return a :class:`DiffZonoBounds` for differential ReLU.

    *x_bounds* and *y_bounds* are standard triangle-relaxation zonotopes.
    For cases 1–8 the diff reuses the **same** epsilon variables as *x* and
    *y*, making ``diff_output = x_output − y_output`` exactly.  For case 9
    (both crossing) a fresh epsilon is introduced for the diff component.

    Examples
    --------
    Active/active regime: diff x-weight equals the relu slope (1.0):

    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.diff.zono3.relu import relu_linearizer
    >>> x = expr.ConstVal(torch.tensor([2.0])) + 0.5 * expr.LpEpsilon([1])
    >>> y = expr.ConstVal(torch.tensor([1.0])) + 0.5 * expr.LpEpsilon([1])
    >>> d = x - y
    >>> dzb = relu_linearizer(x, y, d)
    >>> dzb.x_bounds.input_weights[0].item()
    1.0

    Crossing/crossing regime is still sound for ``relu(x) - relu(y)``:

    >>> x = expr.ConstVal(torch.tensor([0.0])) + 0.8 * expr.LpEpsilon([1])
    >>> y = expr.ConstVal(torch.tensor([0.1])) + 0.8 * expr.LpEpsilon([1])
    >>> d = x - y
    >>> dzb = relu_linearizer(x, y, d)
    >>> dzb.diff_bounds.bias.shape
    torch.Size([1])
    """
    x, y, diff = xs[0], ys[0], ds[0]  # for type checking
    x_ub, x_lb = ublb(x)
    y_ub, y_lb = ublb(y)
    d_ub, d_lb = ublb(diff)
    zeros = torch.zeros_like(x_ub)

    # ------------------------------------------------------------------
    # Standard triangle relaxation for x and y independently
    # ------------------------------------------------------------------
    lam_x = (torch.relu(x_ub) - torch.relu(x_lb)) / (x_ub - x_lb)
    mu_x  = 0.5 * (torch.relu(x_ub) - lam_x * x_ub)

    lam_y = (torch.relu(y_ub) - torch.relu(y_lb)) / (y_ub - y_lb + 1e-30)
    mu_y  = 0.5 * (torch.relu(y_ub) - lam_y * y_ub)
    
    lam_avg = 0.5 * (lam_x + lam_y)
    dx = lam_x - lam_avg
    dy = lam_avg - lam_y
    sd = lam_avg
    bias_d = mu_x - mu_y
    ex = mu_x
    ey = -mu_y
    err_d = torch.zeros_like(ex)

    # ------------------------------------------------------------------
    # Case 9: both x and y crossing → triangle relaxation on d directly
    # ------------------------------------------------------------------
    case9 = (x_ub > 0) & (x_lb < 0) & (y_ub > 0) & (y_lb < 0)
    lam_d = torch.clamp(d_ub / (d_ub - d_lb + 1e-30), 0.0, 1.0)
    nu_d  = lam_d * torch.clamp(-d_lb, min=0.0)
    mu_d  = 0.5 * torch.maximum(d_ub, -d_lb)

    # ------------------------------------------------------------------
    # Diff component — masked per neuron by case9
    #
    # Non-case9: diff = lam_x·x − lam_y·y + (mu_x − mu_y)
    #                   + mu_x·eps_x − mu_y·eps_y   (shared, exact!)
    # Case9:     diff = lam_d·d + (nu_d − mu_d) + mu_d·eps_d  (fresh eps)
    # ------------------------------------------------------------------
    dx     = torch.where(case9, zeros, dx)    # x input weight for diff
    dy     = torch.where(case9, zeros, dy)   # y input weight for diff
    ex     = torch.where(case9, zeros, mu_x)     # scale applied to shared eps_x
    ey     = torch.where(case9, zeros, -mu_y)    # scale applied to shared eps_y
    sd     = torch.where(case9, lam_d, sd)    # d input weight for diff
    bias_d = torch.where(case9, nu_d - mu_d, bias_d)
    err_d  = torch.where(case9, mu_d, err_d)     # fresh error for case-9 neurons


    return DiffZonoBounds(
        x_bounds=ZonoBounds(bias=mu_x, error_coeffs=mu_x, input_weights=[lam_x]),
        y_bounds=ZonoBounds(bias=mu_y, error_coeffs=mu_y, input_weights=[lam_y]),
        diff_bounds=ZonoBounds(
            bias=bias_d,
            error_coeffs=EinsumOp.from_hardmard(err_d, len(x_ub.shape)),
            input_weights=[sd],
        ),
        diff_x_error=EinsumOp.from_hardmard(ex, len(x_ub.shape)),
        diff_x_weights=[dx],
        diff_y_error=EinsumOp.from_hardmard(ey, len(x_ub.shape)),
        diff_y_weights=[dy],
    )
