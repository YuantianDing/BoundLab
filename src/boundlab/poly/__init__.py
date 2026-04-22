r"""Polytope-Based Abstract Interpretation for Neural Networks.

This module provides CROWN-style abstract gates whose semantics are defined
directly by backward-mode bound propagation over linear-polytope relaxations.
"""

from __future__ import annotations

from typing import Literal

import torch

from boundlab.expr._core import Expr, ExprFlags
from boundlab.linearop import LinearOp
from boundlab.linearop._einsum import EinsumOp


class PolyBoundGate(Expr):
    r"""Abstract gate bounded pointwise by a pair of linear polytopes on its child.

    Represents an elementwise function :math:`f(x)` whose output is constrained by

    .. math::
        \lambda_\ell \odot x - 1 \;\le\; f(x) \;\le\; \lambda_u \odot x + 1,

    where :math:`x` is the child expression and :math:`\lambda_u, \lambda_\ell`
    (``upper_lam``, ``lower_lam``) are concrete tensors of the same shape
    as the child.

    Backward propagation splits the incoming weight :math:`w` by sign
    element-wise on its materialized Jacobian. For direction ``"<="``:

    .. math::
        w \cdot f(x) \;\le\; (w_+ \odot \lambda_u + w_- \odot \lambda_\ell)\,x
            + \sum_j |w_{\cdot j}|,

    and symmetrically for ``">="`` with the slopes swapped and the bias negated.
    """

    def __init__(self, child: Expr, upper_lam: torch.Tensor, lower_lam: torch.Tensor):
        super().__init__(ExprFlags.NONE)
        assert child.shape == upper_lam.shape == lower_lam.shape, (
            f"Shape mismatch: child={child.shape}, "
            f"upper_lam={upper_lam.shape}, lower_lam={lower_lam.shape}"
        )
        self._child = child
        self.upper_lam = upper_lam
        self.lower_lam = lower_lam

    @property
    def shape(self) -> torch.Size:
        return self._child.shape

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self._child,)

    def with_children(self, *new_children: Expr) -> "PolyBoundGate":
        (new_child,) = new_children
        return PolyBoundGate(new_child, self.upper_lam, self.lower_lam)

    def backward(self, weights: LinearOp, direction: Literal[">=", "<=", "=="]):
        if direction == "==":
            return None

        jac = weights.jacobian()
        in_ndim = len(weights.input_shape)

        jac_pos = jac.clamp(min=0)
        jac_neg = jac.clamp(max=0)

        if direction == "<=":
            new_jac = jac_pos * self.upper_lam + jac_neg * self.lower_lam
            bias_sign = 1.0
        else:
            new_jac = jac_pos * self.lower_lam + jac_neg * self.upper_lam
            bias_sign = -1.0

        abs_jac = jac.abs()
        if in_ndim > 0:
            reduce_dims = tuple(range(jac.dim() - in_ndim, jac.dim()))
            bias = abs_jac.sum(dim=reduce_dims)
        else:
            bias = abs_jac
        bias = bias_sign * bias

        new_op = EinsumOp.from_full(new_jac, in_ndim)
        return bias, [new_op]

    def to_string(self, child_str: str) -> str:
        return f"PolyBoundGate({child_str})"


__all__ = ["PolyBoundGate"]
