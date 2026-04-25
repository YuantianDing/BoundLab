r"""Softmax handler for polytope abstract interpretation.

Rewrites

.. math::

   \mathrm{softmax}(x)_i
     = \frac{1}{\sum_j \exp(x_j - x_i)}

so that softmax reduces to pairwise subtraction, exp, reduce-sum and
reciprocal — each dispatched through :data:`interpret`.
"""

from boundlab import utils
from boundlab.expr._core import Expr


def softmax_handler(x: Expr, dim: int = -1, dtype=None) -> Expr:
    r"""Polytope softmax transformer via the DeepT decomposition.

    Args:
        x: Input expression.
        dim: Dimension along which to apply softmax (default: -1).
        dtype: Ignored (for API compatibility with :func:`torch.softmax`).

    Returns:
        An expression over-approximating ``torch.softmax(x, dim=dim)``.
    """
    if dim < 0:
        dim += len(x.shape)
    assert dim == len(x.shape) - 1, "softmax_handler only supports the last dimension"

    from . import interpret

    x.simplify_ops_()
    diff = utils.pairwise_diff(x, dim)
    exp_diff = interpret["Exp"](diff)
    sum_exp = exp_diff.sum(dim=-1)
    return interpret["Reciprocal"](sum_exp)
