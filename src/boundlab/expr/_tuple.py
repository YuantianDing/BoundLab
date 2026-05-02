r"""Core Expression Classes

This module defines the base expression class and flags used throughout
the expression framework.
"""

from copy import copy
from typing import Callable, Literal, Union

from numpy import indices
import torch

from boundlab.expr._core import _EXPR_ID_COUNTER, Expr, ExprFlags
from boundlab.linearop._base import LinearOp
    
class TupleExpr:
    """A base class for expressions that represent multiple tensors as a tuple.
    Usually a output of a multi-output operation, such as `torch.chunk` or a convolution that returns both output and pre-activation.

    This is used to represent multiple outputs from a single operation, such
    as the two outputs of a convolution (output and pre-activation). TupleExpr
    is not intended to be used directly by users; it is an implementation
    detail for handling multi-output operations.

    To simplify `Expr` APIs, TupleExpr is not a subclass of `Expr` and does not support the full expression interface.
    It resembles `Expr` in that it has a unique ID and can be used as a child of `Expr` through `GetTupleItem`, 
    but its `shape` is a tuple of `torch.Size`, and its `backward` methods takes multiple weights operators corresponding to each output.

    Attributes:
        children: A tuple of sub-expressions.
    """

    def __init__(self, *flags: ExprFlags):
        self.id = next(_EXPR_ID_COUNTER)
        self.flags = copy(flags)

    @property
    def shape(self) -> tuple[torch.Size, ...]:
        return self._shape

    @property
    def children(self) -> tuple[Expr, ...]:
        """Children expressions that contribute to this TupleExpr. This is used for topological sorting and weight propagation."""
        raise NotImplementedError(f"The :code:`children` property is not implemented for {self.__class__.__name__}.")
    
    def backward(self, *weights, direction = "==") -> tuple[Union[torch.Tensor, Literal[0]], list] | None:
        """Perform backward-mode bound propagation through this expression."""
        raise NotImplementedError(f"The :code:`backward` method is not implemented for {self.__class__.__name__}.")
    
    def with_children(self, *new_children: Expr) -> "TupleExpr":
        """Return a new TupleExpr with the same flags but new children. This is used for expression rewriting during bound propagation."""
        raise NotImplementedError(f"The :code:`with_children` method is not implemented for {self.__class__.__name__}.")
    
    def __getitem__(self, index: int) -> "GetTupleItem":
        """Return a GetTupleItem expression that extracts the specified index from this TupleExpr."""
        return GetTupleItem(self, index)
    
    
class GetTupleItem(Expr):
    """Expression representing indexing into a TupleExpr.

    This is used to extract individual outputs from a TupleExpr. Like TupleExpr,
    this is an implementation detail for handling multi-output operations and
    is not intended for direct use by users.

    Attributes:
        index: The integer index of the tuple element to extract.
        child: The TupleExpr being indexed.
    """

    def __new__(cls, child: Expr, index: int):
        if isinstance(child, MakeTuple):
            # If the child is a MakeTuple, we can directly return the corresponding child expression without creating a GetTupleItem.
            assert 0 <= index < len(child.children), "GetTupleItem index out of range."
            return child.children[index]
        else:
            return super().__new__(cls)

    def __init__(self, child: Expr, index: int):
        super().__init__(ExprFlags.IS_AFFINE)
        assert type(child.shape) is tuple, "GetTupleItem child must be a TupleExpr."
        assert 0 <= index < len(child.children), "GetTupleItem index out of range."
        self._child = child
        self._index = index
        self._shape = child.shape[index]

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def children(self) -> tuple[Expr, ...]:
        raise NotImplementedError("GetTupleItem does not support the children property. It routes the :code:`children` method to :code:`tuple_expr` for topological sorting and weight propagation.")

    @property
    def tuple_expr(self) -> TupleExpr:
        """The TupleExpr being indexed. This is used for topological sorting and weight propagation."""
        return self._child
    
    def simplify_ops_(self):
        for child in self.tuple_expr.children:
            child.simplify_ops_()

    def backward(self, weights, direction = "==") -> tuple[torch.Tensor] | None:
        raise NotImplementedError("GetTupleItem does not support `backward` method. It needs be handled as a special case in bound propagation.")
    
    def with_children(self, *new_children: Expr) -> "GetTupleItem":
        raise NotImplementedError("GetTupleItem does not support the children property. It needs be handled as a special case in bound propagation.")
   
class MakeTuple(TupleExpr):
    """Expression representing the construction of a tuple from multiple sub-expressions.

    This is used to combine multiple expressions into a single TupleExpr, such as the outputs of a multi-output operation. Like GetTupleItem, this is an implementation detail for handling multi-output operations and is not intended for direct use by users.

    Attributes:
        children: A tuple of sub-expressions that are combined into this TupleExpr.
    """

    def __init__(self, *children: Expr):
        super().__init__(*(c.flags for c in children))
        assert all(isinstance(c, Expr) for c in children), "All children of MakeTuple must be Expr instances."
        self._children = children
        self._shape = tuple(c.shape for c in children)

    @property
    def shape(self) -> tuple[torch.Size, ...]:
        return self._shape

    @property
    def children(self) -> tuple[Expr, ...]:
        return self._children

    def backward(self, *weights, direction = "==") -> tuple[torch.Tensor, ...] | None:
        assert len(weights) == len(self.children), "Number of weights must match number of children in MakeTuple."
        return 0, weights
    
    def with_children(self, *new_children: Expr) -> "MakeTuple":
        return MakeTuple(*new_children)
