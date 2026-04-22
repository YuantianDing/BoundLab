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

from boundlab import utils
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
    exp_handler = interpret["Exp"]
    reciprocal_handler = interpret["Reciprocal"]
    ub, lb = x.ublb()
    x.simplify_ops_()

    # pairwise_diff gives diff[..., i, j] = x[..., i] - x[..., j]
    # softmax(x)[i] = 1 / sum_j exp(x[j] - x[i]) = 1 / sum_j exp(-diff[..., i, j])
    diff = utils.pairwise_diff(x, dim)
    expresult = exp_handler(diff)
    assert expresult.ublb()[1].min().item() >= 0, "Expected non-negative lower bound for exponential"
    sum_exp = expresult.sum(dim=dim + 1, keepdim=False)
    print("sum_exp:" + str(sum_exp))
    assert sum_exp.ublb()[1].min().item() >= 0, "Expected non-negative lower bound for sum of exponentials"
    return reciprocal_handler(sum_exp)
