"""

softmax as ``exp → reduce-sum → reciprocal → element-wise product''

"""

import torch

from boundlab.expr._core import Expr
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from .bilinear import diff_bilinear_elementwise


def diff_softmax_handler(x, dim: int = -1, dtype=None):
    r"""Differential softmax transformer.

    When *x* is a :class:`~boundlab.diff.expr.DiffExpr3`, the handler
    decomposes softmax into differential exp, reduce-sum, differential
    reciprocal, and differential element-wise product.

    When *x* is a plain :class:`~boundlab.expr.Expr` or
    :class:`~boundlab.diff.expr.DiffExpr2`, falls back to the
    standard softmax path or promotes to DiffExpr3 first.

    Args:
        x: Input expression or DiffExpr3 with shape ``(m, n)``.
        dim: Softmax dimension (default: -1). Only ``dim=1`` on 2D input
             is currently supported.
        dtype: Ignored (API compatibility).

    Returns:
        Expression or DiffExpr3 over-approximating softmax.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.diff.expr import DiffExpr3
    >>> from boundlab.diff.zono3.softmax import diff_softmax_handler
    >>> x = expr.ConstVal(torch.zeros(2, 3)) + 0.1 * expr.LpEpsilon([2, 3])
    >>> y = expr.ConstVal(torch.ones(2, 3)) + 0.1 * expr.LpEpsilon([2, 3])
    >>> t = DiffExpr3(x, y, x - y)
    >>> out = diff_softmax_handler(t, dim=1)
    >>> out.diff.shape
    torch.Size([2, 3])
    """
    from . import interpret

    if isinstance(x, Expr):
        from boundlab.zono.softmax import softmax_handler as std_softmax
        return std_softmax(x, dim=dim, dtype=dtype)

    if isinstance(x, DiffExpr2):
        x = DiffExpr3(x.x, x.y, x.x - x.y)

    assert isinstance(x, DiffExpr3)

    ndim = len(x.shape)
    if dim < 0:
        dim = ndim + dim

    n = x.shape[dim]

    x_center = x.x.center()
    x_max = x_center.max(dim=dim, keepdim=True).values
    shift = x_max.expand(*x.shape)
    x_shifted = DiffExpr3(
        x.x - shift,
        x.y - shift,
        x.diff,
    )

    exp_handler = interpret["exp"]
    exp_out = exp_handler(x_shifted)

    # Sum along softmax dim using mean * n
    sum_exp = exp_out.mean(dim=dim, keepdim=True) * float(n)

    reciprocal_handler = interpret["reciprocal"]
    inv_sum = reciprocal_handler(sum_exp)

    inv_sum_expanded = inv_sum.expand(*exp_out.shape)

    result = diff_bilinear_elementwise(exp_out, inv_sum_expanded)

    return result


__all__ = ["diff_softmax_handler"]