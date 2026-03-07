

r"""Symbolic Variables for Bound Propagation

This module defines variable expressions representing bounded input
perturbations used in neural network verification.
"""

import math
from typing import Literal, TYPE_CHECKING

import torch

from boundlab.expr._core import Expr, ExprFlags
from boundlab.expr._base import ConstVal

if TYPE_CHECKING:
    pass

class LpEpsilon(Expr):
    r"""A noise symbol bounded by the $\ell_p$-norm constraint.

    Represents a perturbation variable $\boldsymbol{\epsilon}$ satisfying:

    $$\|\boldsymbol{\epsilon}\|_p \leq 1$$

    This is commonly used in zonotope-based verification where each
    $\epsilon_i \in [-1, 1]$ represents an independent noise term.

    During backward propagation with mode ``"<="`` or ``">="``, the
    contribution is computed as $\pm\|\mathbf{w}\|_q$ where $\mathbf{w}$
    is the propagated weight, and $q$ is the dual norm of $p$ defined by $\frac{1}{p} + \frac{1}{q} = 1$.
    """
    def __init__(self, *shape, name=None, p=math.inf):
        super().__init__(ExprFlags.NO_DEPENTENCY)
        self.shape = torch.Size(*shape)
        self.name = name
        self.p = p
        if p == math.inf or p == "inf":
            self.q = 1
        else:
            self.q = 1 / (1 - 1/p) if p > 1 else math.inf

    @property
    def shape(self) -> torch.Size:
        return self.shape
    
    @property
    def children(self) -> tuple[()]:
        return ()
    
    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=","=="]="==") -> tuple[torch.Tensor] | None:
        if mode == "==":
            return None
        assert weights[-len(self.shape):].shape == self.shape, "Incompatible shapes."
        additional_dims = weights.dim() - len(self.shape)
        result = weights.norm(p=self.q, dim=[additional_dims + i for i in range(len(self.shape))])
        if mode == "<=":
            return (result,)
        else:
            return (-result,)
    
    def to_string(self) -> str:
        if self.name is not None:
            return f"𝜀_{self.name}"
        return f"𝜀_{self.id}"

