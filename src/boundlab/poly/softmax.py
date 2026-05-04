"""Softmax handler for polytope abstract interpretation.

Implements softmax through the same DeepT decomposition used by the
zonotope backend:

.. math::

   \mathrm{softmax}(x)_i
     = \frac{1}{\sum_j \exp(x_j - x_i)}

This keeps the transformation in terms of pairwise subtraction,
exponential, reduce-sum and reciprocal.
"""

import torch

from boundlab import utils
from boundlab.expr._core import Expr
from . import _bounds_to_expr
from .reciprocal import reciprocal_linearizer


def softmax_handler(x: Expr, dim: int = -1, dtype=None) -> Expr:
    """Polytope softmax transformer via the DeepT decomposition."""
    if not isinstance(x, Expr):
        return NotImplemented
    if dim < 0:
        dim += len(x.shape)
    assert dim == len(x.shape) - 1, "softmax_handler only supports the last dimension"

    diff = -utils.pairwise_diff(x, dim)
    from . import interpret

    exp_diff = interpret["Exp"](diff)
    sum_exp = exp_diff.sum(dim=-1)
    diff_ub, diff_lb = diff.ublb()

    # Tighten the denominator bounds using the exact interval image of exp.
    sum_exp_ub, sum_exp_lb = sum_exp.ublb()
    exact_sum_ub = torch.exp(diff_ub).sum(dim=-1)
    exact_sum_lb = torch.exp(diff_lb).sum(dim=-1)
    sum_exp_ub = torch.minimum(sum_exp_ub, exact_sum_ub)
    sum_exp_lb = torch.maximum(sum_exp_lb, exact_sum_lb)
    finite_mask = torch.isfinite(sum_exp_ub) & torch.isfinite(sum_exp_lb)
    sum_exp_ub = torch.where(finite_mask, sum_exp_ub, torch.ones_like(sum_exp_ub))
    sum_exp_lb = torch.where(finite_mask, sum_exp_lb, torch.ones_like(sum_exp_lb))
    sum_exp_lb = torch.clamp(sum_exp_lb, min=1e-30)

    bounds = reciprocal_linearizer(sum_exp_ub, sum_exp_lb)
    return _bounds_to_expr(sum_exp, bounds, reason=reciprocal_linearizer.__name__)
