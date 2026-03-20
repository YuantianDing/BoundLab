"""Softmax handler for zonotope abstract interpretation.

Implements softmax as a composed operation:
  softmax(x)_j = exp(x_j) / sum_k exp(x_k)

Following DeepT (Bonaert et al., 2021), uses numerical stabilization
by subtracting the center's max value before applying exp.
"""

import torch

from boundlab.expr._core import Expr
from boundlab.expr._affine import ConstVal
from boundlab.expr._var import LpEpsilon
from .bilinear import bilinear_elementwise


def softmax_handler(x: Expr, dim: int = -1, dtype=None) -> Expr:
    r"""Zonotope softmax transformer built from primitive handlers.

    Softmax is decomposed as:

    .. math::

       \mathrm{softmax}(x)_j = \frac{\exp(x_j)}{\sum_k \exp(x_k)}

    The implementation applies:
    ``exp -> reduce-sum -> reciprocal -> element-wise product``.
    For stability, it first shifts by the center maximum along the softmax
    dimension.
    Currently, only 2D inputs with ``dim == 1`` are supported.

    Args:
        x: Input expression with shape (m, n).
        dim: Dimension along which to apply softmax (default: -1).
        dtype: Ignored (for API compatibility with torch.softmax).

    Returns:
        An expression over-approximating ``torch.softmax(x, dim=dim)``.
    """
    ndim = len(x.shape)
    if dim < 0:
        dim = ndim + dim

    assert ndim == 2 and dim == 1, \
        f"Softmax currently only supports 2D tensors along last dim, got shape {x.shape} dim {dim}"

    n = x.shape[dim]

    # Numerical stability: shift by center's max along softmax dim
    x_center = x.center()
    x_max = x_center.max(dim=dim, keepdim=True).values  # (m, 1)
    x_shifted = x - x_max.expand(*x.shape)  # affine, same shape as x

    # Import the registered handlers from the zonotope interpreter
    from . import interpret
    exp_handler = interpret.dispatcher["exp"]
    reciprocal_handler = interpret.dispatcher["reciprocal"]

    # Apply exp element-wise
    exp_x = exp_handler(x_shifted)

    # Sum along dim 1: exp_x @ ones(n, 1) → (m, 1)
    sum_exp = exp_x @ torch.ones(n, 1)

    # Reciprocal: 1 / sum_exp → (m, 1)
    inv_sum = reciprocal_handler(sum_exp)

    # Broadcast inv_sum to match exp_x shape: (m, 1) → (m, n)
    inv_sum_expanded = inv_sum.expand(*exp_x.shape)

    # Element-wise product: exp_x * inv_sum (bilinear)
    result = bilinear_elementwise(exp_x, inv_sum_expanded)

    return result
