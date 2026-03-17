r"""Bound Propagation for Concretizing Expressions

This module provides functions for computing concrete upper and lower
bounds from symbolic expressions through backward-mode propagation.
"""

import queue
import typing

import torch

import boundlab.expr
from boundlab.linearop import ScalarMul

__all__ = [
    "ub",
    "lb",
    "ublb",
]

if typing.TYPE_CHECKING:
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

_UB_CACHE = {}
_LB_CACHE = {}

def _is0(a) -> bool:
    """Check if a value is effectively zero."""
    from boundlab.linearop import HardmardDot
    if isinstance(a, int) and a == 0:
        return True
    if isinstance(a, HardmardDot) and a.is_zerotensor():
        return True
    return False

def ub(e: "Expr") -> torch.Tensor:
    """Compute an upper bound for the given expression.

    Uses backward-mode bound propagation with direction ``"<="`` to iteratively
    propagate through the expression DAG using a priority queue.

    Args:
        e: The expression to bound.

    Returns:
        A tensor containing the upper bound.
    """
    from boundlab.linearop import HardmardDot
    if e.id in _UB_CACHE:
        return _UB_CACHE[e.id]

    result = torch.zeros(e.shape)

    weight_map = {e.id: HardmardDot.eye(e.shape)}
    pqueue = queue.PriorityQueue()
    pqueue.put(_TopologicalExpr(e))

    while not pqueue.empty():
        current = pqueue.get().expr
        weight = weight_map.pop(current.id)

        backward_result = current.backward(weight, direction="<=")
        if backward_result is None:
            continue

        bias, child_weights = backward_result
        if not _is0(bias):
            result = result + bias

        for child, cw in zip(current.children, child_weights):
            if not _is0(cw):
                if child.id not in weight_map:
                    weight_map[child.id] = cw
                    pqueue.put(_TopologicalExpr(child))
                else:
                    weight_map[child.id] = weight_map[child.id] + cw

    _UB_CACHE[e.id] = result
    return result


def lb(e: "Expr") -> torch.Tensor:
    """Compute a lower bound for the given expression.

    Uses backward-mode bound propagation with direction ``">="`` to iteratively
    propagate through the expression DAG using a priority queue.

    Args:
        e: The expression to bound.

    Returns:
        A tensor containing the lower bound.
    """
    from boundlab.linearop import HardmardDot
    if e.id in _LB_CACHE:
        return _LB_CACHE[e.id]

    result = torch.zeros(e.shape)

    weight_map = {e.id: HardmardDot.eye(e.shape)}
    pqueue = queue.PriorityQueue()
    pqueue.put(_TopologicalExpr(e))

    while not pqueue.empty():
        current = pqueue.get().expr
        weight = weight_map.pop(current.id)

        backward_result = current.backward(weight, direction=">=")
        if backward_result is None:
            continue

        bias, child_weights = backward_result
        if not _is0(bias):
            result = result + bias

        for child, cw in zip(current.children, child_weights):
            if not _is0(cw):
                if child.id not in weight_map:
                    weight_map[child.id] = cw
                    pqueue.put(_TopologicalExpr(child))
                else:
                    weight_map[child.id] = weight_map[child.id] + cw

    _LB_CACHE[e.id] = result
    return result


def ublb(e: "Expr") -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute both upper and lower bounds for the given expression.

    Uses backward propagation in reverse topological order. When an expression
    has the ``SYMMETRIC_TO_0`` flag (e.g. ``LpEpsilon``), its upper bound
    result is reused for the lower bound via negation.

    Args:
        e: The expression to bound.

    Returns:
        A tuple ``(upper_bound, lower_bound)`` of tensors.
    """
    from boundlab.linearop import HardmardDot
    if e.id in _UB_CACHE and e.id in _LB_CACHE:
        return _UB_CACHE[e.id], _LB_CACHE[e.id]

    ub_result = torch.zeros(e.shape)
    lb_result = torch.zeros(e.shape)
    const_result = torch.zeros(e.shape)
    sym_result = torch.zeros(e.shape)

    weight_map = {e.id: ScalarMul(1.0, e.shape)}
    pqueue = queue.PriorityQueue()
    pqueue.put(_TopologicalExpr(e))

    while not pqueue.empty():
        current = pqueue.get().expr
        weight = weight_map.pop(current.id)

        assert weight is not None, (
            f"Missing weight for expression {current.to_string()} (id={current.id}). "
            "This indicates a bug in the bound propagation algorithm."
        )

        child_weights = None

        # Try exact propagation first
        if a := current.backward(weight, direction="=="):
            b, child_weights_exact = a
            if not _is0(b):
                const_result = const_result + b
            child_weights = child_weights_exact

        if child_weights is None:
            if (boundlab.expr.ExprFlags.SYMMETRIC_TO_0 in current.flags
                    and len(current.children) == 0):
                # Leaf symmetric node: compute one-sided bound and reuse via ±
                result = current.backward(weight, direction="<=")
                if result is not None:
                    ubias, _ = result
                    if not _is0(ubias):
                        sym_result = sym_result + ubias
                child_weights = []
            else:
                ub_res = current.backward(weight, direction="<=")
                lb_res = current.backward(weight, direction=">=")
                if ub_res is not None:
                    ubias, uweights = ub_res
                    if not _is0(ubias):
                        ub_result = ub_result + ubias
                else:
                    uweights = []
                if lb_res is not None:
                    lbias, lweights = lb_res
                    if not _is0(lbias):
                        lb_result = lb_result + lbias
                else:
                    lweights = []
                child_weights = list(zip(uweights, lweights))

        for child, weights_pair in zip(current.children, child_weights):
            assert child.id < current.id, (
                f"Child {child.to_string()} (id={child.id}) has higher id than "
                f"parent {current.to_string()} (id={current.id}). Cycle detected."
            )
            if _is0(weights_pair) or weights_pair == (0, 0):
                continue
            if child.id not in weight_map:
                weight_map[child.id] = weights_pair
                pqueue.put(_TopologicalExpr(child))
            else:
                prev = weight_map[child.id]
                if isinstance(weights_pair, tuple):
                    wu, wl = weights_pair
                    if isinstance(prev, tuple):
                        weight_map[child.id] = (prev[0] + wu, prev[1] + wl)
                    else:
                        weight_map[child.id] = (prev + wu, prev + wl)
                else:
                    if isinstance(prev, tuple):
                        weight_map[child.id] = (prev[0] + weights_pair,
                                                 prev[1] + weights_pair)
                    else:
                        weight_map[child.id] = prev + weights_pair

    _UB_CACHE[e.id] = const_result + ub_result + sym_result
    _LB_CACHE[e.id] = const_result + lb_result - sym_result

    return _UB_CACHE[e.id], _LB_CACHE[e.id]


def center(e: "Expr") -> torch.Tensor:
    """Compute the center of the bounds for the given expression."""
    ub_result, lb_result = ublb(e)
    return (ub_result + lb_result) / 2


def bound_width(e: "Expr") -> torch.Tensor:
    """Compute the width of the bounds for the given expression."""
    ub_result, lb_result = ublb(e)
    return ub_result - lb_result
