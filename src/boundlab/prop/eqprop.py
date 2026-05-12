"""Exact propagation helper for symbolic expressions."""

import queue

import torch

import boundlab
from boundlab.linearop._base import LinearOp


def eqprop(x: "boundlab.expr.Expr") -> "boundlab.expr.Expr":
    """Work like `ublb` but stops when `==` propagation is no longer possible, returning an expression."""

    from boundlab.expr import Expr
    from boundlab.expr._affine import AffineSum, ConstVal
    from boundlab.linearop import ScalarOp
    from boundlab.linearop._base import ZeroOp
    # x.simplify_ops_()
    from boundlab.expr._tuple import GetTupleItem, TupleExpr

    subnodes = x.all_subnodes()
    result = ConstVal(x.shape)
    weights = [[ZeroOp(x.shape, s) for s in e.shape] if isinstance(e, TupleExpr) else ZeroOp(x.shape, e.shape) for e in subnodes]
    weights[-1] = ScalarOp(1.0, x.shape)
    

    for i, node in reversed(list(enumerate(subnodes))):
        if isinstance(node, GetTupleItem):
            op = weights[i]
            weights[subnodes.index(node.tuple_expr)][node._index] += op
        elif isinstance(node, TupleExpr):
            ops: tuple[LinearOp, ...] = weights[i]
            if p := node.backward(*ops, direction='=='):
                c, child_ops = p
                for child, child_op in zip(node.children, child_ops):
                    weights[subnodes.index(child)] += child_op
                result = result + c
            else:
                result = result + AffineSum(*[(op, node[i]) for i, op in enumerate(ops)])
        elif isinstance(node, Expr):
            op: LinearOp = weights[i]
            if p := node.backward(op, direction='=='):
                c, child_ops = p
                for child, child_op in zip(node.children, child_ops):
                    weights[subnodes.index(child)] += child_op
                result = result + c
            else:
                result = result + op(node)
        else:
            raise ValueError(f"Unexpected node type {type(node)} in expression DAG.")
    return result
                

            

