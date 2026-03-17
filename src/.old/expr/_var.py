r"""Symbolic Variables for Bound Propagation

This module defines variable expressions representing bounded input
perturbations used in neural network verification.
"""

from typing import Literal, TYPE_CHECKING

import torch

from boundlab.expr._core import Expr, ExprFlags
from boundlab.expr._base import ConstVal

if TYPE_CHECKING:
    pass

class LInfEps(Expr):
    r"""A noise symbol bounded by the $\ell_\infty$-norm constraint.

    Represents a perturbation variable $\boldsymbol{\epsilon}$ satisfying:

    $$\|\boldsymbol{\epsilon}\|_\infty \leq 1$$

    This is commonly used in zonotope-based verification where each
    $\epsilon_i \in [-1, 1]$ represents an independent noise term.

    During backward propagation with mode ``"<="`` or ``">="``, the
    contribution is computed as $\pm\|\mathbf{w}\|_1$ where $\mathbf{w}$
    is the propagated weight, leveraging the duality between $\ell_\infty$
    and $\ell_1$ norms.
    """
    def __init__(self, *shape, name=None):
        super().__init__(ExprFlags.NO_DEPENTENCY)
        self.shape = torch.Size(*shape)
        self.name = name

    @property
    def shape(self) -> torch.Size:
        return self.shape
    
    @property
    def children(self) -> tuple[()]:
        return ()
    
    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=","=="]="==") -> ConstVal | None:
        if mode == "==":
            return None
        assert weights[-len(self.shape):].shape == self.shape, "Incompatible shapes."
        additional_dims = weights.dim() - len(self.shape)
        result = weights.norm(p=1, dim=[additional_dims + i for i in range(len(self.shape))])
        if mode == "<=":
            return ConstVal(result)
        else:
            return ConstVal(-result)
    
    def to_string(self) -> str:
        if self.name is not None:
            return f"𝜀_{self.name}"
        return f"𝜀_{self.id}"

