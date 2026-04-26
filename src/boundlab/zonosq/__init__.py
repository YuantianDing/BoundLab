

from typing import Literal, Union

import torch

from boundlab.expr._core import Expr
from boundlab.expr._tuple import TupleExpr


class LInfEpslionQ(TupleExpr):
    """Represents an L-infinity epsilon term for quantization error."""
    def __init__(self, shape: torch.Size, name=None, reason=None):
        super().__init__(shape)
        self.name = name
        self.reason = reason
    
    @property
    def shape(self) -> tuple[torch.Size, ...]:
        return self._shape

    @property
    def children(self) -> tuple[Expr, ...]:
        """Children expressions that contribute to this TupleExpr. This is used for topological sorting and weight propagation."""
        raise NotImplementedError(f"The :code:`children` property is not implemented for {self.__class__.__name__}.")
    
    def backward(self, weight_lin, weight_sq, direction = "==") -> tuple[torch.Union[torch.Tensor, Literal[0]], list] | None:

    
    def with_children(self, *new_children: Expr) -> "TupleExpr":
        """Return a new TupleExpr with the same flags but new children. This is used for expression rewriting during bound propagation."""
        raise NotImplementedError(f"The :code:`with_children` method is not implemented for {self.__class__.__name__}.")
    
    def to_string(self) -> str:
        if self.name is not None:
            return f"<𝜀 {self.reason} {list(self.shape)}>#{self.name}"
        else:
            return f"<𝜀 {self.reason} {list(self.shape)}>#{self.id}"
    

