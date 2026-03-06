r"""Bound Propagation for Concretizing Expressions

This module provides functions for computing concrete upper and lower
bounds from symbolic expressions through backward-mode propagation.
"""

from typing import TYPE_CHECKING, Iterable

import torch

from boundlab import expr
from boundlab.expr._base import AddList, AddPriorityQueue

if TYPE_CHECKING:
    from boundlab.expr import Expr


def ub(expr: "Expr") -> torch.Tensor:
    """Compute an upper bound for the given expression.

    Args:
        expr: The expression to bound.

    Returns:
        A tensor containing the upper bound.
    """
    # TODO
    pass


def lb(expr: "Expr") -> torch.Tensor:
    """Compute a lower bound for the given expression.

    Args:
        expr: The expression to bound.

    Returns:
        A tensor containing the lower bound.
    """
    # TODO
    pass


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



        
        
