"""

softmax as ``exp → reduce-sum → reciprocal → element-wise product''

"""

import torch

from boundlab.diff.zono3 import expr
from boundlab.expr import Expr, ConstVal
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from .bilinear import diff_bilinear_elementwise


def diff_softmax_handler(x, dim: int = -1, dtype=None, exp_handler=None, reciprocal_handler=None):
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
    >>> from boundlab.diff.zono3.default.softmax import diff_softmax_handler
    >>> x = expr.ConstVal(torch.zeros(2, 3)) + 0.1 * expr.LpEpsilon([2, 3])
    >>> y = expr.ConstVal(torch.ones(2, 3)) + 0.1 * expr.LpEpsilon([2, 3])
    >>> t = DiffExpr3(x, y, x - y)
    >>> out = diff_softmax_handler(t, dim=1)
    >>> out.diff.shape
    torch.Size([2, 3])
    """
    from .. import interpret
    if exp_handler is None:
        exp_handler = interpret["Exp"]
    if reciprocal_handler is None:
        reciprocal_handler = interpret["Reciprocal"]
    
    if isinstance(x, torch.Tensor):
        x = ConstVal(x)
    if isinstance(x, ConstVal):
        return ConstVal(torch.softmax(x.value, dim=dim))
    if isinstance(x, Expr):
        from boundlab.zono.softmax import softmax_handler as std_softmax
        return std_softmax(x, dim=dim, dtype=dtype)

    if isinstance(x, DiffExpr2):
        x = DiffExpr3(x.x, x.y, x.x - x.y)

    assert isinstance(x, DiffExpr3), x

    ndim = len(x.shape)
    if dim < 0:
        dim = ndim + dim

    n = x.shape[dim]

    # DeepT-style rewrite: σ_i(ν) = 1 / Σ_j exp(ν_j - ν_i)
    # Vectorized: reshape ν to (..., N, 1) and (..., 1, N), broadcast-subtract
    # to get an (..., N, N) tensor of ν_j - ν_i, exp, sum over inner dim,
    # reciprocal. Output IS σ. No bilinear needed.

    # Insert a size-1 axis right AFTER dim and right BEFORE dim to create the
    # broadcast shapes. Work with dim as a positive axis.
    # x_i has shape (..., N, 1) — treat as "my row"
    # x_j has shape (..., 1, N) — treat as "all columns"
    x_i = x.unsqueeze(dim + 1)          # shape: (..., N, 1, ...)
    x_j = x.unsqueeze(dim)               # shape: (..., 1, N, ...)

    # BoundLab Expr subtraction requires matching shapes (no auto-broadcast).
    # Expand both to (..., N, N, ...) explicitly before subtract.
    broadcast_shape = list(x.shape)
    broadcast_shape.insert(dim + 1, n)   # add the j axis at dim+1
    x_i_exp = x_i.expand(*broadcast_shape)
    x_j_exp = x_j.expand(*broadcast_shape)

    x_shifted = x_j_exp - x_i_exp

    # exp of the pairwise-difference tensor.
    exp_shifted = exp_handler(x_shifted)

    # Sum along the j-axis (which is now at position dim + 1 since we inserted one
    # new axis at dim + 1 and one at dim, but the j-axis corresponds to dim + 1
    # in the unsqueezed layout). Actually, after x.unsqueeze(dim+1) -> N at dim,
    # 1 at dim+1, then x.unsqueeze(dim) -> 1 at dim, N at dim+1. The SUM we want
    # is over j, which is the axis of size N that comes from x_j — position dim+1.
    sum_exp = exp_shifted.sum(dim=dim + 1, keepdim=False)
    # sum_exp now has shape (..., N, ...) — same as original x.

    # Reciprocal — this IS σ, no bilinear.
    result = reciprocal_handler(sum_exp)

    return result


__all__ = ["diff_softmax_handler"]