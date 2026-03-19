r"""Symbolic Variables for Bound Propagation

This module defines variable expressions representing bounded input
perturbations used in neural network verification.
"""

import math
from typing import Literal

import torch

from boundlab.expr._core import Expr, ExprFlags
from boundlab.linearop._base import LinearOp, LinearOpFlags
from boundlab.linearop._einsum import EinsumOp


class LpEpsilon(Expr):
    r"""A noise symbol bounded by the :math:`\ell_p`-norm constraint.

    Represents a perturbation variable :math:`\boldsymbol{\epsilon}` satisfying:

    .. math:: \|\boldsymbol{\epsilon}\|_p \leq 1

    During backward propagation with direction ``"<="`` or ``">="``, the
    contribution is :math:`\pm\|\mathbf{w}\|_q` where :math:`\mathbf{w}`
    is the materialized weight tensor and :math:`q` is the dual norm of
    :math:`p` defined by :math:`\frac{1}{p} + \frac{1}{q} = 1`.

    Only :class:`~boundlab.linearop.EinsumOp` weights are supported.
    """
    def __init__(self, *shape, name=None, p="inf"):
        super().__init__(ExprFlags.SYMMETRIC_TO_0)
        self._shape = torch.Size(*shape)
        self.name = name
        self.p = p
        if p == math.inf or p == "inf":
            self.q = 1
        else:
            self.q = 1 / (1 - 1/p) if p > 1 else math.inf

    @property
    def shape(self) -> torch.Size:
        return self._shape

    def with_children(self) -> "LpEpsilon":
        return self

    @property
    def children(self) -> tuple[()]:
        return ()

    def backward(self, weights: LinearOp, direction: Literal[">=", "<=", "=="]) \
            -> tuple[torch.Tensor, list] | None:
        r"""Compute the dual-norm bound contribution.

        Args:
            weights: A :class:`~boundlab.linearop.EinsumOp` accumulated
                weight. Must be a ``EinsumOp`` instance.
            direction: ``"<="`` returns :math:`+\|\mathbf{w}\|_q`;
                ``">="`` returns :math:`-\|\mathbf{w}\|_q`;
                ``"=="`` returns ``None``.

        Returns:
            ``(±norm, [])`` or ``None`` for ``"=="``.
        """
        from boundlab.linearop import LinearOp
        if direction == "==":
            return None
        
        result = weights.abs().sum_input()
        assert len(result.input_shape) == 0 and result.output_shape == weights.output_shape
        result = result.jacobian()
        assert result.shape == weights.output_shape
        # input_dims = tuple(range(len(weights.output_shape), len(weights.output_shape) + len(weights.input_shape)))

        # result = weights.force_jacobian().abs().sum(dim=input_dims)
        
        return (result if direction == "<=" else -result, [])

    def to_string(self) -> str:
        if self.name is not None:
            return f"𝜀_{self.name}"
        return f"𝜀_<{self.id:X}>"
