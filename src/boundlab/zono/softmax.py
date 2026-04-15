"""Softmax handler for zonotope abstract interpretation.

Implements softmax following DeepT (Bonaert et al., 2021):

.. math::

   \\mathrm{softmax}(x)_i
     = \\frac{1}{\\sum_j \\exp(x_j - x_i)}

This avoids a bilinear product: only subtraction, exp, reduce-sum
and reciprocal are needed.
"""

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp


def _pairwise_diff(x: Expr, dim: int) -> Expr:
    """Build ``d[..., i, j, ...] = x[..., j, ...] - x[..., i, ...]`` as a
    single :class:`EinsumOp` applied to *x*.

    Using one LinearOp (rather than two broadcasted terms combined via
    subtraction) avoids the ``SumOp`` merging two structurally similar
    :class:`ExpandOp` s that would otherwise cancel the noise contribution.
    """
    N = x.shape[dim]
    l = x.unsqueeze(dim).expand_on(dim, N)
    r = x.unsqueeze(dim + 1).expand_on(dim + 1, N)
    return r - l


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
    ndim = len(x.shape)
    if dim < 0:
        dim = ndim + dim

    from . import interpret
    exp_handler = interpret["Exp"]
    reciprocal_handler = interpret["Reciprocal"]

    # Pairwise differences: diff[..., i, j, ...] = x[..., j, ...] - x[..., i, ...].
    diff = _pairwise_diff(x, dim)

    exp_diff = exp_handler(diff)

    # Sum over j (original softmax axis, now at dim+1) -> shape matches x.
    sum_exp = exp_diff.sum(dim=dim + 1, keepdim=False)

    return reciprocal_handler(sum_exp)
