import torch

import boundlab.expr as expr
import boundlab.prop as prop
from boundlab.expr import Expr, ExprFlags, TupleExpr
from boundlab.expr._tuple import GetTupleItem


class ExactBias(Expr):
    def __init__(self, child: Expr, bias: torch.Tensor):
        super().__init__(ExprFlags.NONE)
        self.child = child
        self.bias = bias

    @property
    def shape(self):
        return self.child.shape

    @property
    def children(self):
        return (self.child,)

    def backward(self, weights, direction="=="):
        if direction != "==":
            return None
        return weights.forward(self.bias), [weights]

    def with_children(self, *new_children):
        return ExactBias(new_children[0], self.bias)


class PairDiffGate(TupleExpr):
    def __init__(self, x: Expr, y: Expr, diff: Expr):
        super().__init__(ExprFlags.NONE, ExprFlags.NONE, ExprFlags.NONE)
        self.x = x
        self.y = y
        self.diff = diff

    @property
    def shape(self):
        return (self.x.shape, self.y.shape)

    @property
    def children(self):
        return (self.x, self.y, self.diff)

    def backward(self, *weights, direction="=="):
        if direction != "==":
            return None
        x_weight, y_weight = weights[:2]
        return 0, [x_weight, y_weight, x_weight - y_weight]

    def with_children(self, *new_children):
        return PairDiffGate(*new_children)


def _reachable(root):
    seen = []

    def visit(node):
        if node in seen:
            return
        seen.append(node)
        if isinstance(node, GetTupleItem):
            for child in node.tuple_expr.children:
                visit(child)
            return
        for child in node.children:
            visit(child)

    visit(root)
    return seen


def test_eqprop_imported_from_prop():
    assert prop.eqprop is not None


def test_eqprop_preserves_zonotope_bounds():
    x = torch.tensor([1.0, -2.0]) + 0.25 * expr.LpEpsilon([2], name="noise")

    rewritten = prop.eqprop(x)

    assert rewritten.shape == x.shape
    assert all(torch.allclose(a, b) for a, b in zip(rewritten.ublb(), x.ublb()))


def test_eqprop_eliminates_exact_expr_node():
    child = expr.LpEpsilon([2], name="child")
    wrapped = ExactBias(child, torch.tensor([1.0, -3.0]))

    rewritten = prop.eqprop(wrapped)

    assert rewritten.shape == wrapped.shape
    assert not any(isinstance(node, ExactBias) for node in _reachable(rewritten))
    assert all(torch.allclose(a, b) for a, b in zip(rewritten.ublb(), wrapped.ublb()))


def test_eqprop_propagates_tuple_equalities():
    x = 0.5 * expr.LpEpsilon([2], name="x")
    y = 0.25 * expr.LpEpsilon([2], name="y")
    diff = 10.0 * expr.LpEpsilon([2], name="diff")
    gate = PairDiffGate(x, y, diff)
    combined = gate[0] + gate[1]

    rewritten = prop.eqprop(combined)
    expected = x + y

    assert rewritten.shape == combined.shape
    assert diff not in _reachable(rewritten)
    assert all(torch.allclose(a, b) for a, b in zip(rewritten.ublb(), expected.ublb()))
