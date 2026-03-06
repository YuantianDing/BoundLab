r"""Bound Propagation for Concretizing Expressions

This module provides functions for computing concrete upper and lower
bounds from symbolic expressions through backward-mode propagation.
"""

from queue import PriorityQueue as _PriorityQueue
from typing import TYPE_CHECKING as _TYPE_CHECKING

import torch as _torch

__all__ = [
    "ub",
    "lb",
    "ublb",
]

from boundlab.utils import eye_of as _eye_of
from boundlab.expr import ExprFlags as _ExprFlags

if _TYPE_CHECKING:
    from boundlab.expr import Expr

class _TopologicalExpr:
    def __init__(self, expr: "Expr"):
        self.expr = expr

    def __eq__(self, other: "_TopologicalExpr") -> bool:
        return -self.expr.id == -other.expr.id
    
    def __lt__(self, other: "_TopologicalExpr") -> int:
        return -self.expr.id < -other.expr.id
    
    def __gt__(self, other: "_TopologicalExpr") -> int:
        return -self.expr.id > -other.expr.id
    
    def __le__(self, other: "_TopologicalExpr") -> int:
        return -self.expr.id <= -other.expr.id
    
    def __ge__(self, other: "_TopologicalExpr") -> int:
        return -self.expr.id >= -other.expr.id
    
    def __ne__(self, other: "_TopologicalExpr") -> int:
        return -self.expr.id != -other.expr.id

def ub(e: "Expr") -> _torch.Tensor:
    """Compute an upper bound for the given expression.

    Uses backward-mode bound propagation with mode ``"<="`` to iteratively
    propagate through the expression DAG using an AddPriorityQueue.

    Args:
        e: The expression to bound.

    Returns:
        A tensor containing the upper bound.
    """
    result = _torch._efficientzerotensor(e.shape)

    weight_map = {e.id: _eye_of(e.shape)}
    queue = _PriorityQueue[_TopologicalExpr]()
    queue.put(_TopologicalExpr(e))

    while not queue.empty():
        current = queue.get().expr
        weight = weight_map.pop(current.id)

        backward_result = current.backward(weight, mode="<=")
        bias = backward_result[0]
        child_weights = backward_result[1:]

        if not _is0(bias):
            result = result + bias

        for child, cw in zip(current.children, child_weights):
            if not _is0(cw):
                if child.id not in weight_map:
                    weight_map[child.id] = cw
                    queue.put(_TopologicalExpr(child))
                else:
                    weight_map[child.id] = weight_map[child.id] + cw

    return result


def lb(e: "Expr") -> _torch.Tensor:
    """Compute a lower bound for the given expression.

    Uses backward-mode bound propagation with mode ``">="`` to iteratively
    propagate through the expression DAG using an AddPriorityQueue.

    Args:
        e: The expression to bound.

    Returns:
        A tensor containing the lower bound.
    """
    result = _torch._efficientzerotensor(e.shape)

    weight_map = {e.id: _eye_of(e.shape)}
    queue = _PriorityQueue[_TopologicalExpr]()
    queue.put(_TopologicalExpr(e))

    while not queue.empty():
        current = queue.get().expr
        weight = weight_map.pop(current.id)

        backward_result = current.backward(weight, mode=">=")
        bias = backward_result[0]
        child_weights = backward_result[1:]

        if not _is0(bias):
            result = result + bias

        for child, cw in zip(current.children, child_weights):
            if not _is0(cw):
                if child.id not in weight_map:
                    weight_map[child.id] = cw
                    queue.put(_TopologicalExpr(child))
                else:
                    weight_map[child.id] = weight_map[child.id] + cw

    return result

def _is0(a):
    return isinstance(a, int) and a == 0

def ublb(e: "Expr") -> tuple[_torch.Tensor, _torch.Tensor]:
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
    ub_result = _torch._efficientzerotensor(e.shape)
    lb_result = _torch._efficientzerotensor(e.shape)
    const_result = _torch._efficientzerotensor(e.shape)
    sym_result = _torch._efficientzerotensor(e.shape)
    
    weight_map = {e.id: _eye_of(e.shape)}
    queue = _PriorityQueue[_TopologicalExpr]()
    queue.put(_TopologicalExpr(e))

    while not queue.empty():
        current = queue.get().expr
        weight = weight_map.pop(current.id)
        child_weights = None
        
        assert weight is not None, f"Missing weight for expression {current.to_string()} (id={current.id}). This indicates a bug in the bound propagation algorithm."
        if isinstance(weight, _torch.Tensor):
            if a := current.backward(weight, mode="=="):
                b, *child_weights = a
                if _is0(b): const_result += b
        if child_weights is None:
            if ExprFlags.SYMMETRIC_TO_0 in current.flags and len(current.children) == 0:
                (ubias,) = current.backward(weight, mode="<=")
                child_weights = ()
                if _is0(ubias): sym_result += ubias
            else:
                ubias, uweights = current.backward(weight, mode="<=")
                lbias, lweights = current.backward(weight, mode=">=")
                if _is0(ubias): ub_result += ubias
                if _is0(lbias): lb_result += lbias
                child_weights = tuple(uweights.zip(lweights))
        
        
        for child, weights in zip(current.children, child_weights):
            assert child.id < current.id, f"Child expression {child.to_string()} (id={child.id}) has higher id than parent {current.to_string()} (id={current.id}). This indicates a cycle in the expression DAG, which should not happen."
            if not _is0(weights) and weights != (0, 0):
                queue.put(_TopologicalExpr(child))
                if child.id not in weight_map:
                    weight_map[child.id] = weights
                else:
                    weight_map[child.id] += weights

    return const_result + ub_result + sym_result, const_result + lb_result - sym_result