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
from boundlab.zono.exp import exp_linearizer
from boundlab.expr._core import Expr

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
