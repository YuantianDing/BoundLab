"""Differential ReLU lineariser (VeryDiff Table 1, 9-case split).

Cases 1–8 reduce to ``relu(x) − relu(y)`` via the standard triangle
relaxation applied independently to *x* and *y*.  For each such case
at most one of (mu_x, mu_y) is non-zero (they occupy disjoint regimes),
so summing the two error terms is lossless.

Case 9 (both crossing) uses a tighter triangle relaxation directly on
``d = x − y`` with slope ``alpha = clamp(d_ub / (d_ub − d_lb), 0, 1)``
and half-width ``mu_d = 0.5 · max(d_ub, −d_lb)``.
"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.relu import relu_linearizer as std_relu_linearizer

from . import _register_linearizer


@_register_linearizer("relu")
def relu_linearizer(
    x: Expr, y: Expr, diff: Expr
) -> tuple[ZonoBounds, ZonoBounds, ZonoBounds]:
    """Return ``(x_bounds, y_bounds, diff_bounds)`` for differential ReLU.

    *x_bounds* and *y_bounds* are standard triangle-relaxation zonotopes.
    *diff_bounds* over-approximates ``relu(x) − relu(y)`` with
    ``input_weights = [sx, sy, sd]`` corresponding to inputs ``[x, y, diff]``.

    Examples
    --------
    Active/active regime: diff weights are ``sx=1, sy=-1``, equivalent to ``x - y``:

    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.diff.zono3.relu import relu_linearizer
    >>> x = expr.ConstVal(torch.tensor([2.0])) + 0.5 * expr.LpEpsilon([1])
    >>> y = expr.ConstVal(torch.tensor([1.0])) + 0.5 * expr.LpEpsilon([1])
    >>> d = x - y
    >>> _, _, d_bounds = relu_linearizer(x, y, d)
    >>> sx, sy = d_bounds.input_weights[0], d_bounds.input_weights[1]
    >>> d_expr = sx * x + sy * y + d_bounds.bias
    >>> torch.allclose(d_expr.ub(), d.ub(), atol=1e-5)
    True

    Crossing/crossing regime is still sound for ``relu(x) - relu(y)``:

    >>> x = expr.ConstVal(torch.tensor([0.0])) + 0.8 * expr.LpEpsilon([1])
    >>> y = expr.ConstVal(torch.tensor([0.1])) + 0.8 * expr.LpEpsilon([1])
    >>> d = x - y
    >>> _, _, d_bounds = relu_linearizer(x, y, d)
    >>> d_bounds.bias.shape
    torch.Size([1])
    """
    x_bounds = std_relu_linearizer(x)
    y_bounds = std_relu_linearizer(y)

    # ------------------------------------------------------------------
    # Cases 1–8: diff ≈ relu(x) − relu(y)
    # ------------------------------------------------------------------
    sx   = x_bounds.input_weights[0]         # slope_x  (0 / 1 / λ_x)
    sy   = -y_bounds.input_weights[0]        # −slope_y
    bias = x_bounds.bias - y_bounds.bias     # μ_x − μ_y
    err  = x_bounds.bias.abs() + y_bounds.bias.abs()  # |μ_x| + |μ_y|

    # ------------------------------------------------------------------
    # Case 9: both crossing → triangle relaxation on d directly
    # ------------------------------------------------------------------
    x_ub, x_lb = ublb(x)
    y_ub, y_lb = ublb(y)
    d_ub, d_lb = ublb(diff)

    x_cross = (x_ub > 0) & (x_lb < 0)
    y_cross = (y_ub > 0) & (y_lb < 0)

    ones  = torch.ones_like(x_ub)
    zeros = torch.zeros_like(x_ub)

    d_denom    = d_ub - d_lb
    degen      = d_denom.abs() < 1e-15
    case9 = x_cross & y_cross & ~degen
    lam      = torch.clamp(d_ub / d_denom, 0.0, 1.0)
    nu         = lam * torch.clamp(-d_lb, min=0.0)
    mu       = 0.5 * torch.maximum(d_ub, -d_lb)

    # Case 9 overrides: zero out x/y slopes, use d directly with lambda slope.
    sx   = torch.where(case9, zeros, sx)
    sy   = torch.where(case9, zeros, sy)
    sd   = torch.where(case9, lam, zeros)
    bias = torch.where(case9, nu - mu, bias)
    err  = torch.where(case9, mu, err)

    # ------------------------------------------------------------------
    # Build the sparse error LinearOp
    # ------------------------------------------------------------------
    output_shape  = x_ub.shape
    error_indices = torch.nonzero(case9, as_tuple=True)
    error_len     = error_indices[0].shape[0]
    error_vals  = err[error_indices]
    indices_op  = SetIndicesOp(error_indices, torch.Size((error_len,)), output_shape)
    hadamard_op = EinsumOp.from_hardmard(error_vals, 1)
    hadamard_op.flags |= LinearOpFlags.IS_NON_NEGATIVE
    error_op = indices_op @ hadamard_op

    diff_bounds = ZonoBounds(bias=bias, error_coeffs=error_op, input_weights=[sx, sy, sd])
    return x_bounds, y_bounds, diff_bounds
