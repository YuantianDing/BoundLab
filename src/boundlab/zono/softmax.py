"""Softmax handler for zonotope abstract interpretation.

Implements softmax following DeepT (Bonaert et al., 2021):

.. math::

   \\mathrm{softmax}(x)_i
     = \\frac{1}{\\sum_j \\exp(x_j - x_i)}

This avoids a bilinear product: only subtraction, exp, reduce-sum
and reciprocal are needed.
"""

from sys import stderr

import torch

from boundlab import expr, utils
from boundlab.expr._var import LpEpsilon
from boundlab.linearop._indices import GatherOp
from boundlab.zono.reciprocal import reciprocal_linearizer
from boundlab.zono.exp import exp_linearizer
from boundlab.expr._core import Expr
from boundlab.zono.softmax2 import softmax2_ibp, softmax2_linearizer
from boundlab.zono.tanh import tanh_linearizer


def softmax_handler(x: Expr, dim: int = -1, dtype=None) -> Expr:
    r"""Zonotope softmax transformer using the DeepT decomposition.

    Softmax is rewritten as:

    .. math::

       \mathrm{softmax}(x)_i
         = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
         = \frac{1}{\sum_j \exp(x_j - x_i)}

    so only subtraction, exp, reduce-sum, and reciprocal are required
    (no bilinear element-wise product).

    Args:
        x: Input expression.
        dim: Dimension along which to apply softmax (default: -1).
        dtype: Ignored (for API compatibility with torch.softmax).

    Returns:
        An expression over-approximating ``torch.softmax(x, dim=dim)``.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.softmax import softmax_handler
    >>> x = expr.ConstVal(torch.zeros(2, 3)) + 0.1 * expr.LpEpsilon([2, 3])
    >>> y = softmax_handler(x, dim=1)
    >>> y.shape
    torch.Size([2, 3])
    """
    if dim < 0:
        dim += len(x.shape)
    assert dim == len(x.shape) - 1, "softmax_handler only supports the last dimension"

    from . import interpret
    ub, lb = x.ublb()

    # pairwise_diff gives diff[..., i, j] = x[..., i] - x[..., j]
    # softmax(x)[i] = 1 / sum_j exp(x[j] - x[i]) = 1 / sum_j exp(diff[..., i, j])
    diff = -utils.pairwise_diff(x, dim)
    ub, lb = diff.ublb()

    expbounds = exp_linearizer(ub, lb)
    bias = expbounds.bias
    error = expbounds.error_coeffs.tensor
    weights = expbounds.input_weights[0]

    finite_mask = torch.isfinite(weights) & torch.isfinite(error) & torch.isfinite(bias) & (lb < 30) & (ub < 30)
    bias = torch.where(finite_mask, bias, 0)
    error = torch.where(finite_mask, error, 0)
    weights = torch.where(finite_mask, weights, 0)
    sum_exp = (weights * diff).sum(dim=-1) + error.sum(dim=-1) * LpEpsilon(diff.shape[:-2]) + bias.sum(dim=-1)
    finite_mask = finite_mask.all(dim=-1)
    
    sum_exp_ub, sum_exp_lb = sum_exp.ublb()
    sum_exp_ub = torch.minimum(sum_exp_ub, torch.exp(ub).sum(dim=-1))
    sum_exp_lb = torch.maximum(sum_exp_lb, torch.exp(lb).sum(dim=-1))
    bounds = reciprocal_linearizer(sum_exp_ub, sum_exp_lb)
    w = bounds.input_weights[0]
    mu = bounds.bias
    beta = bounds.error_coeffs.tensor
    result = finite_mask * (w * sum_exp + mu + beta * LpEpsilon(sum_exp.shape))
    return result


def softmax_handler_basedon_softmax2(x: Expr, dim: int = -1, dtype=None) -> Expr:
    if dim < 0:
        dim += len(x.shape)
    assert dim == len(x.shape) - 1, "softmax_handler_basedon_softmax2 only supports the last dimension"

    # Pairwise differences for recurrence y = x_j - x_i.
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    diff = -utils.pairwise_diff(x, dim=-1) # [T, i, j]
    diff = diff.permute(1, 2, 0) # [i, j, T]
    diff = utils.remove_diagonal(diff, dim1=0, dim2=1) # [i, j, T]
    diff = diff.permute(1, 2, 0) # [j, T, i]
    diff_ub, diff_lb = diff.ublb()

    tanh_err = (diff_ub + diff_lb).abs() / 2

    indices = torch.argsort(tanh_err, dim=0)
    diff_ub = torch.gather(diff_ub, dim=0, index=indices)
    diff_lb = torch.gather(diff_lb, dim=0, index=indices)
    diff = GatherOp(diff.shape, dim=0, index=indices)(diff)

    result = expr.ConstVal(torch.ones(x.shape))
    result_ub, result_lb = result.ublb()
    for i in range(diff.shape[0]):
        # Domain invariant of this recurrence: result stays in (0, 1].
        result_ub = torch.clamp(result_ub, min=1e-8, max=1.0)
        result_lb = torch.clamp(result_lb, min=1e-8, max=1.0)
        result_lb = torch.minimum(result_lb, result_ub)

        zonobounds = softmax2_linearizer(result_ub, result_lb, diff_ub[i], diff_lb[i])
        ub_ibp, lb_ibp = softmax2_ibp(result_ub, result_lb, diff_ub[i], diff_lb[i])

        lam_x, lam_y, mu, beta = zonobounds.input_weights[0], zonobounds.input_weights[1], zonobounds.bias, zonobounds.error_coeffs.tensor
        result = lam_x * result + lam_y * diff[i] + mu + beta * LpEpsilon(result.shape)
        result_ub, result_lb = result.ublb()
        result_ub = torch.minimum(result_ub, ub_ibp)
        result_lb = torch.maximum(result_lb, lb_ibp)

    return result.reshape(orig_shape)
