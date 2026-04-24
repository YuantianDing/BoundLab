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
    x.simplify_ops_()

    # pairwise_diff gives diff[..., i, j] = x[..., i] - x[..., j]
    # softmax(x)[i] = 1 / sum_j exp(x[j] - x[i]) = 1 / sum_j exp(diff[..., i, j])
    diff = utils.pairwise_diff(x, dim)
    ub, lb = diff.ublb()

    expbounds = exp_linearizer(ub, lb)
    bias = expbounds.bias
    error = expbounds.error_coeffs.tensor
    weights = expbounds.input_weights[0]
    print(weights.shape, diff.shape)

    # finite_mask = torch.isfinite(weights) & torch.isfinite(error) & torch.isfinite(bias) & (lb < 20) & (ub < 20)
    # bias = torch.where(finite_mask, bias, 1)
    # error = torch.where(finite_mask, error, 0)
    # weights = torch.where(finite_mask, weights, 0)
    # assert (weights * lb - error + bias >= -1e-8).all(), f"Softmax denominator has non-positive lower bound, which should be impossible {(weights * lb - error + bias).min()}"
    exp_exp = weights * diff + error * LpEpsilon(diff.shape[:-2]) + bias
    # assert exp_exp.ublb()[1].min() >= -1e-8, f"Softmax denominator has non-positive lower bound, which should be impossible {exp_exp.ublb()[1].min()}"
    sum_exp = (weights * diff).sum(dim=-1) + error.sum(dim=-1) * LpEpsilon(diff.shape[:-2]) + bias.sum(dim=-1)
    # assert sum_exp.ublb()[0].min() >= 0, "Softmax output has negative lower bound, which should be impossible"
    # finite_mask = finite_mask.all(dim=-1)
    result = interpret["Reciprocal"](sum_exp)
    return result
