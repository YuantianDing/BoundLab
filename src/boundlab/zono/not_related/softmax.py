"""Softmax handler for zonotope abstract interpretation.

Implements softmax following DeepT (Bonaert et al., 2021):

.. math::

   \\mathrm{softmax}(x)_i
     = \\frac{1}{\\sum_j \\exp(x_j - x_i)}

This avoids a bilinear product: only subtraction, exp, reduce-sum
and reciprocal are needed.
"""

from boundlab import expr

import torch

from boundlab import utils
from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.zono.softmax2 import softmax2_handler




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
    assert dim == -1 or dim == len(x.shape) - 1

    from . import interpret
    exp_handler = interpret["Exp"]
    reciprocal_handler = interpret["Reciprocal"]

    # Pairwise differences: diff[..., i, j, ...] = x[..., j, ...] - x[..., i, ...].
    x = x.reshape(-1, x.shape[-1])
    diff = utils.pairwise_diff(x, dim=-1)
    diff = diff.permute(1, 2, 0)
    diff = utils.remove_diagonal(diff, dim1=0, dim2=1)
    diff = diff.permute(1, 2, 0)

    result = expr.ConstVal(torch.ones(x.shape))
    for i in range(diff.shape[0]):
        result = interpret["Softmax2"](result, diff[i])
        print(result.bound_width().max().item())

    return result
