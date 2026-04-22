r"""Bound Propagation for Concretizing Expressions

This module provides functions for computing concrete upper and lower
bounds from symbolic expressions through backward-mode propagation.

Examples
--------
>>> import torch
>>> import boundlab.expr as expr
>>> import boundlab.prop as prop
>>> x = expr.ConstVal(torch.tensor([1.0, -1.0])) + expr.LpEpsilon([2])
>>> ub = prop.ub(x)
>>> lb = prop.lb(x)
>>> ub.shape, lb.shape
(torch.Size([2]), torch.Size([2]))
"""

import queue
import typing

import torch

import boundlab.expr
from boundlab.linearop import ScalarOp
from boundlab.linearop._base import ZeroOp

__all__ = [
    "ub",
    "lb",
    "ublb",
]

if typing.TYPE_CHECKING:
    from boundlab.expr import Expr

class _TopologicalExpr:
    def __init__(self, expr):
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
    from boundlab.linearop import EinsumOp
    if isinstance(a, int) and a == 0:
        return True
    if isinstance(a, ZeroOp):
        return True
    return False


def _accumulate_tuple_weight(tuple_weight_map, pqueue, te, idx, weight):
    """Accumulate a weight for a TupleExpr at a given index."""
    if te.id not in tuple_weight_map:
        tuple_weight_map[te.id] = {}
        pqueue.put(_TopologicalExpr(te))
    d = tuple_weight_map[te.id]
    if idx in d:
        d[idx] = d[idx] + weight
    else:
        d[idx] = weight


def _propagate_to_children(weight_map, pqueue, children, child_weights):
    """Propagate child weights into weight_map and enqueue new children."""
    for child, cw in zip(children, child_weights):
        if not _is0(cw):
            if child.id not in weight_map:
                weight_map[child.id] = cw
                pqueue.put(_TopologicalExpr(child))
            else:
                weight_map[child.id] = weight_map[child.id] + cw


def ub(e: "Expr") -> torch.Tensor:
    r"""Compute an upper bound via backward bound propagation.

    This function propagates linear weights backward through the expression DAG
    in direction ``"<="`` and accumulates resulting bias terms.

    Args:
        e: The expression to bound.

    Returns:
        A tensor :math:`u` such that :math:`x \le u` for all concrete values
        represented by ``e``.

    Notes:
        Results are memoized in ``_UB_CACHE`` keyed by expression id.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> x = expr.ConstVal(torch.tensor([0.0])) + expr.LpEpsilon([1])
    >>> ub(x).shape
    torch.Size([1])
    """
    from boundlab.linearop import EinsumOp
    from boundlab.expr._tuple import GetTupleItem, TupleExpr

    if e.id in _UB_CACHE:
        return _UB_CACHE[e.id]

    result = torch.zeros(e.shape)

    weight_map = {e.id: ScalarOp(1.0, e.shape)}
    tuple_weight_map = {}
    pqueue = queue.PriorityQueue()
    pqueue.put(_TopologicalExpr(e))

    while not pqueue.empty():
        current = pqueue.get().expr

        if isinstance(current, GetTupleItem):
            weight = weight_map.pop(current.id)
            _accumulate_tuple_weight(tuple_weight_map, pqueue,
                                     current.tuple_expr, current._index, weight)
            continue

        if isinstance(current, TupleExpr):
            wd = tuple_weight_map.pop(current.id)
            ws = [wd.get(i, 0) for i in range(len(current.children))]
            backward_result = current.backward(*ws, direction="<=")
            if backward_result is not None:
                bias, child_weights = backward_result
                if not _is0(bias):
                    result = result + bias
                _propagate_to_children(weight_map, pqueue,
                                       current.children, child_weights)
            continue

        weight = weight_map.pop(current.id)

        backward_result = current.backward(weight, direction="<=")
        if backward_result is None:
            continue

        bias, child_weights = backward_result
        if not _is0(bias):
            result = result + bias

        _propagate_to_children(weight_map, pqueue,
                               current.children, child_weights)

    _UB_CACHE[e.id] = result
    return result


def lb(e: "Expr") -> torch.Tensor:
    r"""Compute a lower bound via backward bound propagation.

    This function propagates linear weights backward through the expression DAG
    in direction ``">="`` and accumulates resulting bias terms.

    Args:
        e: The expression to bound.

    Returns:
        A tensor :math:`l` such that :math:`x \ge l` for all concrete values
        represented by ``e``.

    Notes:
        Results are memoized in ``_LB_CACHE`` keyed by expression id.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> x = expr.ConstVal(torch.tensor([0.0])) + expr.LpEpsilon([1])
    >>> lb(x).shape
    torch.Size([1])
    """
    from boundlab.linearop import EinsumOp
    from boundlab.expr._tuple import GetTupleItem, TupleExpr

    if e.id in _LB_CACHE:
        return _LB_CACHE[e.id]

    result = torch.zeros(e.shape)

    weight_map = {e.id: ScalarOp(1.0, e.shape)}
    tuple_weight_map = {}
    pqueue = queue.PriorityQueue()
    pqueue.put(_TopologicalExpr(e))

    while not pqueue.empty():
        current = pqueue.get().expr

        if isinstance(current, GetTupleItem):
            weight = weight_map.pop(current.id)
            _accumulate_tuple_weight(tuple_weight_map, pqueue,
                                     current.tuple_expr, current._index, weight)
            continue

        if isinstance(current, TupleExpr):
            wd = tuple_weight_map.pop(current.id)
            ws = [wd.get(i, 0) for i in range(len(current.children))]
            backward_result = current.backward(*ws, direction=">=")
            if backward_result is not None:
                bias, child_weights = backward_result
                if not _is0(bias):
                    result = result + bias
                _propagate_to_children(weight_map, pqueue,
                                       current.children, child_weights)
            continue

        weight = weight_map.pop(current.id)

        backward_result = current.backward(weight, direction=">=")
        if backward_result is None:
            continue

        bias, child_weights = backward_result
        if not _is0(bias):
            result = result + bias

        _propagate_to_children(weight_map, pqueue,
                               current.children, child_weights)

    _LB_CACHE[e.id] = result
    return result


def _ublb_add_weight(prev, new):
    """Add two ublb weights, each either a single LinearOp or a (ub, lb) tuple.
    Preserves single form when both inputs are single."""
    prev_is_tuple = isinstance(prev, tuple)
    new_is_tuple = isinstance(new, tuple)
    if not prev_is_tuple and not new_is_tuple:
        return prev + new
    pu, pl = prev if prev_is_tuple else (prev, prev)
    nu, nl = new if new_is_tuple else (new, new)
    return (pu + nu, pl + nl)


def _ublb_propagate_children(weight_map, pqueue, children, child_weights):
    """Propagate child weights for ublb (handles both single and tuple weights)."""
    for child, weights_pair in zip(children, child_weights):
        if _is0(weights_pair) or weights_pair == (0, 0):
            continue
        if child.id not in weight_map:
            weight_map[child.id] = weights_pair
            pqueue.put(_TopologicalExpr(child))
        else:
            weight_map[child.id] = _ublb_add_weight(weight_map[child.id], weights_pair)


def _ublb_split_results(ub_res, lb_res):
    """Unpack split-mode backward results into (ub_bias, lb_bias, child_weight_pairs)."""
    if ub_res is not None:
        ubias, uweights = ub_res
    else:
        ubias, uweights = 0, []
    if lb_res is not None:
        lbias, lweights = lb_res
    else:
        lbias, lweights = 0, []
    return ubias, lbias, list(zip(uweights, lweights))


def ublb(e: "Expr") -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute both upper and lower bounds for the given expression.

    Uses backward propagation in reverse topological order. When an expression
    has the ``SYMMETRIC_TO_0`` flag (e.g. ``LpEpsilon``), its upper bound
    result is reused for the lower bound via negation.

    Args:
        e: The expression to bound.

    Returns:
        A tuple ``(upper_bound, lower_bound)``.

    Notes:
        For symmetric leaf expressions (flag ``SYMMETRIC_TO_0``), only one
        side needs to be computed; the opposite side is obtained by negation.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> x = expr.ConstVal(torch.tensor([2.0])) + expr.LpEpsilon([1])
    >>> u, l = ublb(x)
    >>> (u >= l).all().item()
    True
    """
    from boundlab.linearop import EinsumOp
    from boundlab.expr._tuple import GetTupleItem, TupleExpr

    if e.id in _UB_CACHE and e.id in _LB_CACHE:
        return _UB_CACHE[e.id], _LB_CACHE[e.id]

    ub_result = torch.zeros(e.shape)
    lb_result = torch.zeros(e.shape)
    const_result = torch.zeros(e.shape)
    sym_result = torch.zeros(e.shape)

    weight_map = {e.id: ScalarOp(1.0, e.shape)}
    tuple_weight_map = {}
    pqueue = queue.PriorityQueue()
    pqueue.put(_TopologicalExpr(e))

    while not pqueue.empty():
        current = pqueue.get().expr

        # Handle GetTupleItem: route weight to its TupleExpr
        if isinstance(current, GetTupleItem):
            weight = weight_map.pop(current.id)
            _accumulate_tuple_weight(tuple_weight_map, pqueue,
                                     current.tuple_expr, current._index, weight)
            continue

        # Handle TupleExpr: backward with per-index weights
        if isinstance(current, TupleExpr):
            wd = tuple_weight_map.pop(current.id)
            n = len(current.children)
            ws = [wd.get(i, 0) for i in range(n)]

            all_single = all(not isinstance(w, tuple) for w in ws)
            child_weights = None

            if all_single:
                if a := current.backward(*ws, direction="=="):
                    b, cw_exact = a
                    if not _is0(b):
                        const_result = const_result + b
                    child_weights = list(cw_exact)

            if child_weights is None:
                ub_ws = [w[0] if isinstance(w, tuple) else w for w in ws]
                lb_ws = [w[1] if isinstance(w, tuple) else w for w in ws]
                ubias, lbias, child_weights = _ublb_split_results(
                    current.backward(*ub_ws, direction="<="),
                    current.backward(*lb_ws, direction=">="),
                )
                if not _is0(ubias):
                    ub_result = ub_result + ubias
                if not _is0(lbias):
                    lb_result = lb_result + lbias

            _ublb_propagate_children(weight_map, pqueue,
                                     current.children, child_weights)
            continue

        weight = weight_map.pop(current.id)

        assert weight is not None, (
            f"Missing weight for expression {current.to_string()} (id={current.id}). "
            "This indicates a bug in the bound propagation algorithm."
        )

        child_weights = None
        is_split = isinstance(weight, tuple)

        # Try exact propagation first (only valid for single weights)
        if not is_split:
            if a := current.backward(weight, direction="=="):
                b, child_weights_exact = a
                if not _is0(b):
                    const_result = const_result + b
                child_weights = child_weights_exact

        if child_weights is None:
            if (not is_split
                    and current.flags & boundlab.expr.ExprFlags.SYMMETRIC_TO_0 != 0
                    and len(current.children) == 0):
                # Leaf symmetric node: compute one-sided bound and reuse via ±
                result = current.backward(weight, direction="<=")
                if result is not None:
                    ubias, _ = result
                    if not _is0(ubias):
                        sym_result = sym_result + ubias
                child_weights = []
            else:
                u_w, l_w = weight if is_split else (weight, weight)
                ubias, lbias, child_weights = _ublb_split_results(
                    current.backward(u_w, direction="<="),
                    current.backward(l_w, direction=">="),
                )
                if not _is0(ubias):
                    ub_result = ub_result + ubias
                if not _is0(lbias):
                    lb_result = lb_result + lbias

        _ublb_propagate_children(weight_map, pqueue,
                                 current.children, child_weights)

    _UB_CACHE[e.id] = const_result + ub_result + sym_result
    _LB_CACHE[e.id] = const_result + lb_result - sym_result

    return _UB_CACHE[e.id], _LB_CACHE[e.id]


def center(e: "Expr") -> torch.Tensor:
    r"""Compute the midpoint of the concretized interval.

    .. math::

       \mathrm{center}(e) = \frac{\mathrm{ub}(e) + \mathrm{lb}(e)}{2}
    """
    ub_result, lb_result = ublb(e)
    return (ub_result + lb_result) / 2


def bound_width(e: "Expr") -> torch.Tensor:
    r"""Compute interval width from concretized bounds.

    .. math::

       \mathrm{width}(e) = \mathrm{ub}(e) - \mathrm{lb}(e)
    """
    ub_result, lb_result = ublb(e)
    return ub_result - lb_result
