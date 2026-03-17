r"""Bound Propagation for Concretizing Expressions

This module provides functions for computing concrete upper and lower
bounds from symbolic expressions through backward-mode propagation.
"""

from typing import TYPE_CHECKING, Iterable, Literal

import torch

from boundlab import expr
from boundlab.expr._base import AddList, AddPriorityQueue

if TYPE_CHECKING:
    from boundlab.expr import Expr


def ub(e: "Expr") -> torch.Tensor:
    """Compute an upper bound for the given expression.

    Uses backward-mode bound propagation with mode ``"<="`` to iteratively
    propagate through the expression DAG using an AddPriorityQueue.

    Args:
        e: The expression to bound.

    Returns:
        A tensor containing the upper bound.
    """
    return _bound(e, mode="<=")


def lb(e: "Expr") -> torch.Tensor:
    """Compute a lower bound for the given expression.

    Uses backward-mode bound propagation with mode ``">="`` to iteratively
    propagate through the expression DAG using an AddPriorityQueue.

    Args:
        e: The expression to bound.

    Returns:
        A tensor containing the lower bound.
    """
    return _bound(e, mode=">=")


def _bound(e: "Expr", mode: Literal["<=", ">="]) -> torch.Tensor:
    """Internal helper for computing bounds via backward propagation.

    Follows the same iterative pattern as ublb, using AddPriorityQueue
    to process terms and backward propagation to reach leaf expressions.

    Args:
        e: The expression to bound.
        mode: ``"<="`` for upper bound, ``">="`` for lower bound.

    Returns:
        A tensor containing the computed bound.
    """
    # Handle ConstTensorDot: propagate backward through child
    if isinstance(e, expr.ConstTensorDot):
        if e2 := e.child.backward(e.value, mode=mode):
            return _bound(e2, mode)
        else:
            # Child doesn't support backward with this mode
            # This shouldn't happen for well-formed expressions
            raise NotImplementedError(
                f"Child {type(e.child).__name__} doesn't support backward with mode={mode}"
            )
    elif isinstance(e, expr.ConstVal):
        return e.value
    # Handle non-Add expressions: use backward_eye
    elif not isinstance(e, expr.Add):
        if e2 := e.backward_eye(mode=mode):
            return _bound(e2, mode)
        else:
            # Leaf expression that returns None from backward_eye
            # For ConstVal, backward_eye returns a ConstVal
            if isinstance(e, expr.ConstVal):
                return e.value
            # For other leaves, no contribution
            return torch.zeros(e.shape)

    # Handle Add: use priority queue to iteratively process children
    children = AddPriorityQueue(e.children)
    result = torch.zeros(e.shape)

    while len(children) > 0:
        child = children.pop()

        if len(child.children) == 0:
            # Leaf expression: extract its bound contribution
            if isinstance(child, expr.ConstVal):
                result = result + child.value
            else:
                # Other leaf (e.g., LInfEps) - use backward_eye to get bound
                if e2 := child.backward_eye(mode=mode):
                    result = result + _bound(e2, mode)
                # If backward_eye returns None, no contribution
                else:
                    raise NotImplementedError(
                        f"Leaf expression {type(child).__name__} doesn't support backward_eye with mode={mode}"
                    )
        elif isinstance(child, expr.ConstTensorDot):
            # Propagate backward through the child
            if e2 := child.child.backward(child.value, mode=mode):
                children.add(e2)
            else:
                # Child doesn't support backward with this mode
                # This shouldn't happen for well-formed expressions
                raise NotImplementedError(
                    f"Child {type(child.child).__name__} of ConstTensorDot doesn't support backward with mode={mode}"
                )
        else:
            # Other expression types: use backward_eye
            if e2 := child.backward_eye(mode=mode):
                children.add(e2)
            else:
                raise NotImplementedError(
                    f"Expression {type(child).__name__} doesn't support backward_eye with mode={mode}"
                )

    return result


def ublb(e: "Expr") -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute both upper and lower bounds for the given expression.

    This is achieved by iteratively applying backward propagation from the
    output to the inputs in reverse topological order. When an expression
    has the ``SYMMETRIC_TO_0`` flag set, both bounds are computed
    simultaneously for efficiency.

    Args:
        e: The expression to bound.

    Returns:
        A tuple ``(upper_bound, lower_bound)`` of tensors.
    """
    if e.flags & expr.ExprFlags.SYMMETRIC_TO_0:
        u = ub(e)
        return u, -u
    elif isinstance(e, expr.ConstTensorDot):
        if e2 := e.child.backward(e.value, mode="=="):
            return ublb(e2)
        else:
            return ub(e), lb(e)
    elif isinstance(e, expr.ConstVal):
        return e.value, e.value
    elif not isinstance(e, expr.Add):
        if e2 := e.backward_eye(mode="=="):
            return ublb(e2)
        else:
            return ub(e), lb(e)
        
    children = AddPriorityQueue(e.children)
    symmetrics = AddList()
    result_ub = torch.zeros(e.shape)
    result_lb = torch.zeros(e.shape)
    while child := children.pop():
        if len(child.children) == 0:
            if child.flags & expr.ExprFlags.SYMMETRIC_TO_0:
                symmetrics.append(child)
            else:
                result_ub += ub(child)
                result_lb += lb(child)
        elif isinstance(child, expr.ConstTensorDot):
            if e2 := child.child.backward(child.value, mode="=="):
                children.add(e2)
            else:
                l1, l2 = _split(child, children)
                expr_new = expr.Add(*l1)
                result_ub += ub(expr_new)
                result_lb += lb(expr_new)
                children = AddPriorityQueue(l2)
        else:
            if e2 := child.backward_eye(mode="=="):
                children.add(e2)
            else:
                l1, l2 = _split(child, children)
                expr_new = expr.Add(*l1)
                result_ub += ub(expr_new)
                result_lb += lb(expr_new)
                children = AddPriorityQueue(l2)
                
    U = ub(expr.Add(*symmetrics.terms))
    result_ub += U
    result_lb -= U
    return result_ub, result_lb

def _split(expr: Expr, children: Iterable[expr.Expr]):
    leave_ids = expr.get_all_leave_ids()
    l1 = []
    l2 = []
    for child in children:
        if leave_ids.intersection(child.get_all_leave_ids()):
            l1.append(child)
        else:            
            l2.append(child)
    return l1, l2



        
        
