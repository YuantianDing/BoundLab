import torch
import pytest

import boundlab.expr as expr
import boundlab.prop as prop
from boundlab.expr._tuple import TupleExpr, GetTupleItem, MakeTuple
from boundlab.expr._core import ExprFlags


# ---------------------------------------------------------------------------
# A non-MakeTuple TupleExpr for testing GetTupleItem (since MakeTuple's
# __new__ short-circuits GetTupleItem creation).
# ---------------------------------------------------------------------------

class PassthroughTuple(TupleExpr):
    """TupleExpr that passes weights through to children, like MakeTuple
    but without the __new__ short-circuit on GetTupleItem."""

    def __init__(self, *children):
        super().__init__(ExprFlags.IS_AFFINE)
        assert all(isinstance(c, expr.Expr) for c in children)
        self._children = children
        self._shape = tuple(c.shape for c in children)

    @property
    def children(self):
        return self._children

    def backward(self, *weights, direction="=="):
        assert len(weights) == len(self.children)
        return 0, weights

    def with_children(self, *new_children):
        return PassthroughTuple(*new_children)


# ---------------------------------------------------------------------------
# MakeTuple basic tests
# ---------------------------------------------------------------------------

def test_make_tuple_shape():
    a = expr.LpEpsilon([3])
    b = expr.LpEpsilon([4])
    t = MakeTuple(a, b)
    assert t.shape == (torch.Size([3]), torch.Size([4]))


def test_make_tuple_children():
    a = expr.LpEpsilon([2])
    b = expr.LpEpsilon([3])
    t = MakeTuple(a, b)
    assert t.children == (a, b)


def test_make_tuple_has_id():
    t1 = MakeTuple(expr.LpEpsilon([1]))
    t2 = MakeTuple(expr.LpEpsilon([1]))
    assert hasattr(t1, 'id')
    assert t1.id != t2.id


def test_make_tuple_getitem_short_circuits():
    """MakeTuple[i] returns the child directly (no GetTupleItem created)."""
    a = expr.LpEpsilon([2])
    b = expr.LpEpsilon([3])
    t = MakeTuple(a, b)
    assert t[0] is a
    assert t[1] is b


def test_make_tuple_with_children():
    a = expr.LpEpsilon([2])
    b = expr.LpEpsilon([3])
    t = MakeTuple(a, b)
    c = expr.LpEpsilon([2])
    d = expr.LpEpsilon([3])
    t2 = t.with_children(c, d)
    assert t2.children == (c, d)


# ---------------------------------------------------------------------------
# GetTupleItem basic tests (using PassthroughTuple)
# ---------------------------------------------------------------------------

def test_get_tuple_item_shape():
    a = expr.LpEpsilon([2])
    b = expr.LpEpsilon([5])
    t = PassthroughTuple(a, b)
    item0 = t[0]
    item1 = t[1]
    assert isinstance(item0, GetTupleItem)
    assert isinstance(item1, GetTupleItem)
    assert item0.shape == torch.Size([2])
    assert item1.shape == torch.Size([5])


def test_get_tuple_item_tuple_expr():
    a = expr.LpEpsilon([3])
    t = PassthroughTuple(a)
    item = t[0]
    assert item.tuple_expr is t


def test_get_tuple_item_index_out_of_range():
    a = expr.LpEpsilon([2])
    t = PassthroughTuple(a)
    with pytest.raises(AssertionError):
        t[1]


# ---------------------------------------------------------------------------
# Bound propagation through GetTupleItem + PassthroughTuple
# ---------------------------------------------------------------------------

def test_get_tuple_item_ub_lb_epsilon():
    """Bound propagation through GetTupleItem with LpEpsilon children."""
    a = expr.LpEpsilon([2])
    b = expr.LpEpsilon([3])
    t = PassthroughTuple(a, b)
    item_a = t[0]
    item_b = t[1]
    assert torch.allclose(prop.ub(item_a), torch.ones(2))
    assert torch.allclose(prop.lb(item_a), -torch.ones(2))
    assert torch.allclose(prop.ub(item_b), torch.ones(3))
    assert torch.allclose(prop.lb(item_b), -torch.ones(3))


def test_get_tuple_item_ub_lb_const():
    """Bound propagation through GetTupleItem with ConstVal children."""
    a = expr.ConstVal(torch.tensor([1.0, 2.0]))
    b = expr.ConstVal(torch.tensor([3.0, 4.0, 5.0]))
    t = PassthroughTuple(a, b)
    item_a = t[0]
    item_b = t[1]
    assert torch.allclose(prop.ub(item_a), torch.tensor([1.0, 2.0]))
    assert torch.allclose(prop.lb(item_a), torch.tensor([1.0, 2.0]))
    assert torch.allclose(prop.ub(item_b), torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(prop.lb(item_b), torch.tensor([3.0, 4.0, 5.0]))


def test_get_tuple_item_ublb():
    """ublb through GetTupleItem."""
    eps = expr.LpEpsilon([2])
    c = expr.ConstVal(torch.tensor([1.0, -1.0]))
    t = PassthroughTuple(c + eps, c - eps)
    x = t[0]  # c + eps
    y = t[1]  # c - eps

    ub_x, lb_x = prop.ublb(x)
    assert torch.allclose(ub_x, torch.tensor([2.0, 0.0]))
    assert torch.allclose(lb_x, torch.tensor([0.0, -2.0]))

    ub_y, lb_y = prop.ublb(y)
    assert torch.allclose(ub_y, torch.tensor([2.0, 0.0]))
    assert torch.allclose(lb_y, torch.tensor([0.0, -2.0]))


def test_get_tuple_item_arithmetic():
    """Arithmetic on GetTupleItem results, then bound propagation."""
    eps = expr.LpEpsilon([2])
    c = expr.ConstVal(torch.ones(2))
    t = PassthroughTuple(c + eps, c - eps)
    x = t[0]  # c + eps: [0, 2]
    y = t[1]  # c - eps: [0, 2]

    # x + y = 2c (eps cancels out)
    z = x + y
    ub_z, lb_z = prop.ublb(z)
    assert torch.allclose(ub_z, torch.tensor([2.0, 2.0]))
    assert torch.allclose(lb_z, torch.tensor([2.0, 2.0]))


def test_get_tuple_item_shared_children():
    """Multiple GetTupleItems from the same TupleExpr share structure."""
    eps = expr.LpEpsilon([3])
    t = PassthroughTuple(eps, eps)
    x = t[0]
    y = t[1]
    # x - y should be exactly 0 since both point to same eps
    diff = x - y
    ub_d, lb_d = prop.ublb(diff)
    assert torch.allclose(ub_d, torch.zeros(3), atol=1e-6)
    assert torch.allclose(lb_d, torch.zeros(3), atol=1e-6)


def test_get_tuple_item_partial_use():
    """Only one element of a tuple is used; others should not affect bounds."""
    a = expr.LpEpsilon([2])
    b = expr.LpEpsilon([3])
    t = PassthroughTuple(a, b)
    # Only use item 0
    item = t[0]
    assert torch.allclose(prop.ub(item), torch.ones(2))
    assert torch.allclose(prop.lb(item), -torch.ones(2))


def test_get_tuple_item_scaled():
    """Scaled GetTupleItem results."""
    eps = expr.LpEpsilon([2])
    t = PassthroughTuple(eps)
    x = t[0] * 2.0
    assert torch.allclose(prop.ub(x), torch.tensor([2.0, 2.0]))
    assert torch.allclose(prop.lb(x), torch.tensor([-2.0, -2.0]))


# ---------------------------------------------------------------------------
# MakeTuple bound propagation (short-circuited, so no GetTupleItem involved)
# ---------------------------------------------------------------------------

def test_make_tuple_bounds_via_indexing():
    """MakeTuple[i] short-circuits to child, so bounds are direct."""
    eps = expr.LpEpsilon([2])
    c = expr.ConstVal(torch.tensor([1.0, -1.0]))
    t = MakeTuple(eps, c)
    # t[0] is eps directly, t[1] is c directly
    item0 = t[0]
    item1 = t[1]
    assert torch.allclose(prop.ub(item0), torch.ones(2))
    assert torch.allclose(prop.lb(item0), -torch.ones(2))
    assert torch.allclose(prop.ub(item1), torch.tensor([1.0, -1.0]))
    assert torch.allclose(prop.lb(item1), torch.tensor([1.0, -1.0]))
