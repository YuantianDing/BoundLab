import torch
import pytest

import boundlab
import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.expr._cat import Cat, Stack


def test_version():
    assert boundlab.__version__ == "0.1.0"


def test_torch_available():
    assert torch.__version__


# ---------------------------------------------------------------------------
# LpEpsilon
# ---------------------------------------------------------------------------

def test_lp_epsilon_shape():
    eps = expr.LpEpsilon([4])
    assert eps.shape == torch.Size([4])


def test_lp_epsilon_shape_named():
    eps = expr.LpEpsilon([3], name="noise")
    assert eps.shape == torch.Size([3])
    assert "noise" in eps.to_string()


def test_lp_epsilon_linf_ub_lb():
    eps = expr.LpEpsilon([3])
    assert torch.allclose(eps.ub(), torch.ones(3))
    assert torch.allclose(eps.lb(), -torch.ones(3))


def test_lp_epsilon_with_children():
    eps = expr.LpEpsilon([2])
    assert eps.with_children() is eps


# ---------------------------------------------------------------------------
# ConstVal
# ---------------------------------------------------------------------------

def test_const_val_bounds():
    c = expr.ConstVal(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(c.ub(), torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(c.lb(), torch.tensor([1.0, 2.0, 3.0]))


def test_const_val_to_string():
    c = expr.ConstVal(torch.zeros(1), name="bias")
    assert c.to_string() == "#const bias"


# ---------------------------------------------------------------------------
# Add: ConstVal + LpEpsilon
# ---------------------------------------------------------------------------

def test_add_shifted_bounds():
    center = expr.ConstVal(torch.tensor([0.5, -1.0]))
    eps = expr.LpEpsilon([2])
    x = center + eps
    assert torch.allclose(x.ub(), torch.tensor([1.5, 0.0]))
    assert torch.allclose(x.lb(), torch.tensor([-0.5, -2.0]))


def test_add_three_children():
    a = expr.ConstVal(torch.ones(2))
    b = expr.LpEpsilon([2])
    c = expr.LpEpsilon([2], name="c2")
    x = a + b + c
    # center=1, two independent eps each in [-1,1]: ub=3, lb=-1
    assert torch.allclose(x.ub(), torch.tensor([3.0, 3.0]))
    assert torch.allclose(x.lb(), torch.tensor([-1.0, -1.0]))


# ---------------------------------------------------------------------------
# ublb consistency with ub / lb
# ---------------------------------------------------------------------------

def test_ublb_consistency():
    center = expr.ConstVal(torch.tensor([2.0, -1.0]))
    eps = expr.LpEpsilon([2])
    x = center + eps
    ub_val, lb_val = x.ublb()
    assert torch.allclose(ub_val, x.ub())
    assert torch.allclose(lb_val, x.lb())


# ---------------------------------------------------------------------------
# Cat
# ---------------------------------------------------------------------------

def test_cat_shape():
    a = expr.LpEpsilon([3])
    b = expr.LpEpsilon([5])
    c = Cat(a, b, dim=0)
    assert c.shape == torch.Size([8])


def test_cat_bounds():
    a = expr.LpEpsilon([2])
    b = expr.LpEpsilon([3])
    c = Cat(a, b, dim=0)
    assert torch.allclose(c.ub(), torch.ones(5))
    assert torch.allclose(c.lb(), -torch.ones(5))


def test_cat_2d_shape():
    # Bounds need N-D eye_of (not yet implemented); shape only.
    a = expr.LpEpsilon([2, 4])
    b = expr.LpEpsilon([3, 4])
    c = Cat(a, b, dim=0)
    assert c.shape == torch.Size([5, 4])


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

def test_stack_shape():
    a = expr.LpEpsilon([4])
    b = expr.LpEpsilon([4])
    s = Stack(a, b, dim=0)
    assert s.shape == torch.Size([2, 4])


def test_stack_shape_with_bounds_skipped():
    # Stack output is 2D; bounds need N-D eye_of (not yet implemented).
    a = expr.LpEpsilon([3])
    b = expr.LpEpsilon([3])
    s = Stack(a, b, dim=0)
    assert s.shape == torch.Size([2, 3])


def test_stack_dim1():
    a = expr.LpEpsilon([3])
    b = expr.LpEpsilon([3])
    s = Stack(a, b, dim=1)
    assert s.shape == torch.Size([3, 2])


# ---------------------------------------------------------------------------
# ReLU linearizer
# ---------------------------------------------------------------------------
# Call the registered handler directly: zono.interpret["relu"]

def test_relu_dead_neuron():
    # center=-2, eps in [-1,1] -> ub=-1 <= 0: dead neuron, output is 0
    center = expr.ConstVal(torch.tensor([-2.0]))
    eps = expr.LpEpsilon([1])
    x = center + eps
    handler = zono.interpret["relu"]
    result = handler(x)
    assert torch.allclose(result.ub(), torch.zeros(1), atol=1e-6)
    assert torch.allclose(result.lb(), torch.zeros(1), atol=1e-6)


def test_relu_active_neuron():
    # center=2, eps in [-1,1] -> lb=1 >= 0: active, identity pass-through
    center = expr.ConstVal(torch.tensor([2.0]))
    eps = expr.LpEpsilon([1])
    x = center + eps
    handler = zono.interpret["relu"]
    result = handler(x)
    assert torch.allclose(result.ub(), torch.tensor([3.0]), atol=1e-5)
    assert torch.allclose(result.lb(), torch.tensor([1.0]), atol=1e-5)


def test_relu_crossing_neuron():
    # center=0, eps in [-1,1] -> lb=-1, ub=1: crossing
    # slope = 1/(1-(-1)) = 0.5, bias = error = -1*(-1)/(2*2) = 0.25
    # output = 0.5*x + 0.25 + 0.25*eps_new
    # ub = 0.5*1 + 0.25 + 0.25*1 = 1.0
    # lb = 0.5*(-1) + 0.25 + 0.25*(-1) = -0.5
    eps = expr.LpEpsilon([1])
    handler = zono.interpret["relu"]
    result = handler(eps)
    assert result.ub().item() == pytest.approx(1.0, abs=1e-5)
    assert result.lb().item() == pytest.approx(-0.5, abs=1e-5)
