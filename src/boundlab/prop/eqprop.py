from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from boundlab.expr._core import Expr


def eqprop(x: "Expr") -> "Expr":
    """Propagate equality constraints on an expression.

    This is a no-op for zonotopes, but can be used to propagate equalities
    in other abstract domains. e.g. ZonoHexGate
    """
    from boundlab.expr._affine import AffineSum, ConstVal
    from boundlab.expr._tuple import GetTupleItem, TupleExpr
    from boundlab.linearop import ScalarOp
    from boundlab.prop import (
        _TopologicalExpr,
        _accumulate_tuple_weight,
        _is0,
        _propagate_to_children,
    )
    import queue

    x.simplify_ops_()

    const = torch.zeros(x.shape)
    terms = []
    weight_map = {x.id: ScalarOp(1.0, x.shape)}
    tuple_weight_map = {}
    pqueue = queue.PriorityQueue()
    pqueue.put(_TopologicalExpr(x))

    def add_boundary(weight, expr):
        if _is0(weight):
            return
        rewritten_children = tuple(eqprop(child) for child in expr.children)
        rewritten = expr.with_children(*rewritten_children) if rewritten_children else expr
        terms.append((weight, rewritten))

    def add_tuple_boundary(tuple_expr, weights):
        rewritten_children = tuple(eqprop(child) for child in tuple_expr.children)
        rewritten_tuple = tuple_expr.with_children(*rewritten_children)
        for index, weight in enumerate(weights):
            if index >= len(tuple_expr.shape) or _is0(weight):
                continue
            terms.append((weight, GetTupleItem(rewritten_tuple, index)))

    while not pqueue.empty():
        current = pqueue.get().expr

        if isinstance(current, GetTupleItem):
            weight = weight_map.pop(current.id)
            _accumulate_tuple_weight(
                tuple_weight_map,
                pqueue,
                current.tuple_expr,
                current._index,
                weight,
            )
            continue

        if isinstance(current, TupleExpr):
            wd = tuple_weight_map.pop(current.id)
            weights = [wd.get(i, 0) for i in range(len(current.children))]
            if all(not isinstance(weight, tuple) for weight in weights):
                backward_result = current.backward(*weights, direction="==")
                if backward_result is not None:
                    bias, child_weights = backward_result
                    if not _is0(bias):
                        const = const + bias
                    _propagate_to_children(
                        weight_map,
                        pqueue,
                        current.children,
                        child_weights,
                    )
                    continue
            add_tuple_boundary(current, weights)
            continue

        weight = weight_map.pop(current.id)
        backward_result = current.backward(weight, direction="==")
        if backward_result is not None:
            bias, child_weights = backward_result
            if not _is0(bias):
                const = const + bias
            _propagate_to_children(weight_map, pqueue, current.children, child_weights)
            continue

        add_boundary(weight, current)

    if terms:
        return AffineSum(*terms, const=const)
    return ConstVal(const)
