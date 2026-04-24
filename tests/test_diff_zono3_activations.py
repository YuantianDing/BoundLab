"""Soundness tests for differential tanh, exp, reciprocal, bilinear, and softmax.

For each nonlinearity we:
1. Build a DiffExpr3 ``(x, y, d)`` with independent epsilons and ``d = x - y``.
2. Run the triple through the differential handler.
3. Sample many concrete pairs from the L∞ perturbation balls.
4. Assert every concrete difference ``f(s1) - f(s2)`` lies within the
   computed bounds of the output diff component.
"""

import torch
import pytest
from torch import nn

import boundlab.expr as expr
from boundlab.diff.zono3.expr import DiffExpr2, DiffExpr3
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export

def _export(model: nn.Module, *in_shapes: list[int]):
    return onnx_export(model, in_shapes)


def _zonotope(center: torch.Tensor, scale: float = 1.0) -> expr.Expr:
    e = expr.LpEpsilon(list(center.shape))
    return center + scale * e


def _make_triple(c1: torch.Tensor, c2: torch.Tensor, scale: float = 1.0) -> DiffExpr3:
    x = _zonotope(c1, scale)
    y = _zonotope(c2, scale)
    return DiffExpr3(x, y, x - y)


def _sample_pairs(c1, c2, n: int = 2000, scale: float = 1.0):
    s1 = c1 + (torch.rand(n, *c1.shape) * 2 - 1) * scale
    s2 = c2 + (torch.rand(n, *c2.shape) * 2 - 1) * scale
    return s1, s2


def _check_sound(d_expr: expr.Expr, diffs: torch.Tensor, tol: float = 1e-5):
    d_ub, d_lb = d_expr.ublb()
    assert (diffs <= d_ub.unsqueeze(0) + tol).all(), (
        f"UB violated: max excess = {(diffs - d_ub.unsqueeze(0)).max():.6f}"
    )
    assert (diffs >= d_lb.unsqueeze(0) - tol).all(), (
        f"LB violated: max deficit = {(d_lb.unsqueeze(0) - diffs).max():.6f}"
    )

def _const_zonotope(center_val: float, half_width: float, n: int = 1) -> expr.Expr:
    c = torch.full((n,), center_val)
    return expr.Add(expr.ConstVal(c), torch.full((n,), half_width) * expr.LpEpsilon([n]))

_tanh_handler = diff_interpret["tanh"]

def test_tanh_diff_identical_inputs_sound():
    """When x == y, tanh diff bounds contain zero (sound)."""
    c = torch.tensor([0.5, -0.3, 1.0])
    x = _zonotope(c, scale=0.2)
    out = _tanh_handler(DiffExpr3(x, x, x - x))
    d_ub, d_lb = out.diff.ublb()
    assert (d_ub >= -1e-6).all(), f"UB must be >= 0: {d_ub}"
    assert (d_lb <= 1e-6).all(), f"LB must be <= 0: {d_lb}"

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_tanh_diff_sound(seed: int):
    """Tanh diff bounds contain all sampled tanh(x) - tanh(y)."""
    torch.manual_seed(seed)
    n = 6
    c1 = torch.randn(n) * 0.5
    c2 = torch.randn(n) * 0.5
    scale = 0.5

    triple = _make_triple(c1, c2, scale)
    out = _tanh_handler(triple)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    _check_sound(out.diff, torch.tanh(s1) - torch.tanh(s2))

def test_tanh_diff_large_input_sound():
    """Tanh diff soundness for larger input ranges (saturation regime)."""
    torch.manual_seed(20)
    c1 = torch.tensor([2.0, -2.0, 0.0, 3.0])
    c2 = torch.tensor([-1.0, 1.0, 0.5, -3.0])
    scale = 1.0

    triple = _make_triple(c1, c2, scale)
    out = _tanh_handler(triple)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    _check_sound(out.diff, torch.tanh(s1) - torch.tanh(s2))

def test_tanh_diff_diffexpr2_promotes():
    """Tanh on DiffExpr2 promotes to DiffExpr3."""
    x = _const_zonotope(0.5, 0.3)
    y = _const_zonotope(-0.2, 0.3)
    pair = DiffExpr2(x, y)
    out = _tanh_handler(pair)
    assert isinstance(out, DiffExpr3)

def test_tanh_diff_fallback_plain_expr():
    """Plain Expr through tanh handler matches standard zonotope."""
    import boundlab.zono as zono
    torch.manual_seed(30)
    x = _zonotope(torch.randn(4), scale=0.5)
    out_diff = _tanh_handler(x)
    std_handler = zono.interpret["tanh"]
    out_std = std_handler(x)
    assert torch.allclose(out_diff.ub(), out_std.ub(), atol=1e-6)
    assert torch.allclose(out_diff.lb(), out_std.lb(), atol=1e-6)

_exp_handler = diff_interpret["exp"]

def test_exp_diff_identical_inputs_sound():
    """When x == y, exp diff bounds contain zero (sound)."""
    c = torch.tensor([0.5, -0.3, 1.0])
    x = _zonotope(c, scale=0.2)
    out = _exp_handler(DiffExpr3(x, x, x - x))
    d_ub, d_lb = out.diff.ublb()
    assert (d_ub >= -1e-6).all(), f"UB must be >= 0: {d_ub}"
    assert (d_lb <= 1e-6).all(), f"LB must be <= 0: {d_lb}"

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_exp_diff_sound(seed: int):
    """Exp diff bounds contain all sampled exp(x) - exp(y)."""
    torch.manual_seed(seed)
    n = 6
    c1 = torch.randn(n) * 0.5
    c2 = torch.randn(n) * 0.5
    scale = 0.3

    triple = _make_triple(c1, c2, scale)
    out = _exp_handler(triple)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    _check_sound(out.diff, torch.exp(s1) - torch.exp(s2))

def test_exp_diff_large_input_sound():
    """Exp diff soundness for larger inputs."""
    torch.manual_seed(20)
    c1 = torch.tensor([1.0, -1.0, 0.5, 2.0])
    c2 = torch.tensor([0.5, 0.0, -0.5, 1.0])
    scale = 0.5

    triple = _make_triple(c1, c2, scale)
    out = _exp_handler(triple)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    _check_sound(out.diff, torch.exp(s1) - torch.exp(s2))
def test_exp_diff_fallback_plain_expr():
    """Plain Expr through exp handler matches standard zonotope."""
    import boundlab.zono as zono
    torch.manual_seed(30)
    x = _zonotope(torch.randn(4) * 0.5, scale=0.3)
    out_diff = _exp_handler(x)
    std_handler = zono.interpret["exp"]
    out_std = std_handler(x)
    assert torch.allclose(out_diff.ub(), out_std.ub(), atol=1e-6)
    assert torch.allclose(out_diff.lb(), out_std.lb(), atol=1e-6)

_reciprocal_handler = diff_interpret["reciprocal"]

def test_reciprocal_diff_identical_inputs_sound():
    """When x == y (both positive), reciprocal diff bounds contain zero."""
    c = torch.tensor([2.0, 3.0, 1.5])
    x = _zonotope(c, scale=0.2)
    out = _reciprocal_handler(DiffExpr3(x, x, x - x))
    d_ub, d_lb = out.diff.ublb()
    assert (d_ub >= -1e-6).all(), f"UB must be >= 0: {d_ub}"
    assert (d_lb <= 1e-6).all(), f"LB must be <= 0: {d_lb}"
@pytest.mark.parametrize("seed", [10, 11, 12])
def test_reciprocal_diff_sound(seed: int):
    """Reciprocal diff bounds contain all sampled 1/x - 1/y."""
    torch.manual_seed(seed)
    n = 6
    # Ensure strictly positive inputs
    c1 = torch.rand(n) * 2 + 1.0   # in [1, 3]
    c2 = torch.rand(n) * 2 + 1.0
    scale = 0.3

    triple = _make_triple(c1, c2, scale)
    out = _reciprocal_handler(triple)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    _check_sound(out.diff, 1.0 / s1 - 1.0 / s2)


def test_reciprocal_diff_large_positive_sound():
    """Reciprocal diff soundness for larger positive values."""
    torch.manual_seed(20)
    c1 = torch.tensor([5.0, 2.0, 10.0])
    c2 = torch.tensor([3.0, 4.0, 8.0])
    scale = 0.5

    triple = _make_triple(c1, c2, scale)
    out = _reciprocal_handler(triple)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    _check_sound(out.diff, 1.0 / s1 - 1.0 / s2)

def test_reciprocal_diff_fallback_plain_expr():
    """Plain Expr through reciprocal handler matches standard zonotope."""
    import boundlab.zono as zono
    torch.manual_seed(30)
    c = torch.rand(4) * 2 + 1.0
    x = _zonotope(c, scale=0.2)
    out_diff = _reciprocal_handler(x)
    std_handler = zono.interpret["reciprocal"]
    out_std = std_handler(x)
    assert torch.allclose(out_diff.ub(), out_std.ub(), atol=1e-6)
    assert torch.allclose(out_diff.lb(), out_std.lb(), atol=1e-6)

from boundlab.diff.zono3.default.bilinear import (
    diff_bilinear_elementwise,
    diff_bilinear_matmul,
    diff_mul_handler,
    diff_matmul_handler,
)

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_bilinear_elementwise_sound(seed: int):
    """Element-wise product diff bounds contain all sampled a1*b1 - a2*b2."""
    torch.manual_seed(seed)
    n = 5
    ca1, ca2 = torch.randn(n), torch.randn(n)
    cb1, cb2 = torch.randn(n), torch.randn(n)
    scale = 0.3

    a = _make_triple(ca1, ca2, scale)
    b = _make_triple(cb1, cb2, scale)
    out = diff_bilinear_elementwise(a, b)

    sa1, sa2 = _sample_pairs(ca1, ca2, scale=scale, n=3000)
    sb1, sb2 = _sample_pairs(cb1, cb2, scale=scale, n=3000)
    _check_sound(out.diff, sa1 * sb1 - sa2 * sb2, tol=1e-4)

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_bilinear_matmul_sound(seed: int):
    """Matmul diff bounds contain all sampled A1@B1 - A2@B2."""
    torch.manual_seed(seed)
    m, k, n_out = 2, 3, 2
    ca1 = torch.randn(m, k)
    ca2 = torch.randn(m, k)
    cb1 = torch.randn(k, n_out)
    cb2 = torch.randn(k, n_out)
    scale = 0.2

    a = _make_triple(ca1, ca2, scale)
    b = _make_triple(cb1, cb2, scale)
    out = diff_bilinear_matmul(a, b)

    N = 3000
    sa1 = ca1 + (torch.rand(N, m, k) * 2 - 1) * scale
    sa2 = ca2 + (torch.rand(N, m, k) * 2 - 1) * scale
    sb1 = cb1 + (torch.rand(N, k, n_out) * 2 - 1) * scale
    sb2 = cb2 + (torch.rand(N, k, n_out) * 2 - 1) * scale
    conc = torch.bmm(sa1, sb1) - torch.bmm(sa2, sb2)
    _check_sound(out.diff, conc, tol=1e-4)
def test_bilinear_elementwise_identical_is_zero():
    """When a==b and both branches identical, product diff is zero."""
    c = torch.tensor([1.0, 2.0, 3.0])
    x = _zonotope(c, scale=0.2)
    a = DiffExpr3(x, x, x - x)
    b = DiffExpr3(x, x, x - x)
    out = diff_bilinear_elementwise(a, b)
    d_ub, d_lb = out.diff.ublb()
    assert torch.allclose(d_ub, torch.zeros(3), atol=1e-4)
    assert torch.allclose(d_lb, torch.zeros(3), atol=1e-4)
def test_diff_mul_handler_diffexpr3_times_scalar():
    """DiffExpr3 * scalar uses __mul__, not bilinear."""
    c1, c2 = torch.randn(3), torch.randn(3)
    t = _make_triple(c1, c2)
    out = diff_mul_handler(t, 2.0)
    assert isinstance(out, (DiffExpr3, DiffExpr2))
    d_ub, d_lb = out.diff.ublb()
    t_ub, t_lb = t.diff.ublb()
    assert torch.allclose(d_ub, t_ub * 2.0, atol=1e-6)
def test_diff_mul_handler_diffexpr3_times_tensor():
    """DiffExpr3 * Tensor uses __mul__."""
    c1, c2 = torch.randn(3), torch.randn(3)
    t = _make_triple(c1, c2)
    w = torch.tensor([1.0, -1.0, 0.5])
    out = diff_mul_handler(t, w)
    assert isinstance(out, (DiffExpr3, DiffExpr2))
def test_diff_matmul_handler_fallback():
    """Plain Expr @ Tensor falls back to standard matmul."""
    torch.manual_seed(40)
    x = _zonotope(torch.randn(4))
    W = torch.randn(3, 4)
    out = diff_matmul_handler(W, x)
    assert isinstance(out, expr.Expr)
    assert out.shape == torch.Size([3])

from boundlab.diff.zono3.default.softmax import diff_softmax_handler

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_softmax_diff_sound(seed: int):
    """Softmax diff bounds contain all sampled softmax(x) - softmax(y)."""
    torch.manual_seed(seed)
    m, n = 2, 3
    c1 = torch.randn(m, n) * 0.3
    c2 = torch.randn(m, n) * 0.3
    scale = 0.1

    triple = _make_triple(c1, c2, scale)
    out = diff_softmax_handler(triple, dim=1)

    N = 3000
    s1 = c1 + (torch.rand(N, m, n) * 2 - 1) * scale
    s2 = c2 + (torch.rand(N, m, n) * 2 - 1) * scale
    conc = torch.softmax(s1, dim=-1) - torch.softmax(s2, dim=-1)
    _check_sound(out.diff, conc, tol=1e-3)

def test_softmax_diff_identical_small():
    """When x == y, softmax diff should be near zero."""
    c = torch.randn(2, 3) * 0.3
    x = _zonotope(c, scale=0.1)
    triple = DiffExpr3(x, x, x - x)
    out = diff_softmax_handler(triple, dim=1)
    d_ub, d_lb = out.diff.ublb()
    # Not exactly zero due to bilinear approximation, but should be small
    assert (d_ub >= -1e-3).all()
    assert (d_lb <= 1e-3).all()

def test_softmax_diff_fallback_plain_expr():
    """Plain Expr through softmax handler matches standard zonotope."""
    import boundlab.zono as zono
    torch.manual_seed(30)
    x = _zonotope(torch.randn(2, 3) * 0.3, scale=0.1)
    out_diff = diff_softmax_handler(x, dim=1)
    std_handler = zono.interpret["Softmax"]
    out_std = std_handler(x, axis=1)
    assert torch.allclose(out_diff.ub(), out_std.ub(), atol=1e-6)
    assert torch.allclose(out_diff.lb(), out_std.lb(), atol=1e-6)

def test_softmax_diff_diffexpr2_promotes():
    """Softmax on DiffExpr2 promotes to DiffExpr3."""
    c1, c2 = torch.randn(2, 3) * 0.3, torch.randn(2, 3) * 0.3
    x, y = _zonotope(c1, 0.1), _zonotope(c2, 0.1)
    pair = DiffExpr2(x, y)
    out = diff_softmax_handler(pair, dim=1)
    assert isinstance(out, DiffExpr3)

def test_interpreter_tanh_mlp_diff_sound():
    """MLP with Tanh: diff bounds are sound."""
    torch.manual_seed(50)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3))
    c1, c2 = torch.randn(4), torch.randn(4)
    op = diff_interpret(_export(model, [4]))
    out = op(_make_triple(c1, c2))
    s1, s2 = _sample_pairs(c1, c2)
    with torch.no_grad():
        _check_sound(out.diff, model(s1) - model(s2))


@pytest.mark.parametrize("seed", [60, 61, 62])
def test_interpreter_deep_tanh_mlp_diff_sound(seed: int):
    """Deep MLP with Tanh: diff bounds are sound."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(5, 8), nn.Tanh(),
        nn.Linear(8, 6), nn.Tanh(),
        nn.Linear(6, 3),
    )
    c1, c2 = torch.randn(5), torch.randn(5)
    op = diff_interpret(_export(model, [5]))
    out = op(_make_triple(c1, c2))
    s1, s2 = _sample_pairs(c1, c2, n=2500)
    with torch.no_grad():
        _check_sound(out.diff, model(s1) - model(s2), tol=2e-5)

def test_tanh_fallback_mlp_matches_std():
    """Plain Expr through Tanh MLP matches standard zonotope interpreter."""
    import boundlab.zono as zono
    torch.manual_seed(70)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3))
    gm = _export(model, [4])
    x = _zonotope(torch.randn(4))

    y_diff = diff_interpret(gm)(x)
    y_std = zono.interpret(gm)(x)

    assert torch.allclose(y_diff.ub(), y_std.ub(), atol=1e-5)
    assert torch.allclose(y_diff.lb(), y_std.lb(), atol=1e-5)
def test_tanh_diff_tighter_than_independent():
    """Shared eps: tanh diff bounds are tighter than naive subtraction."""
    torch.manual_seed(80)
    import boundlab.zono as zono
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3))
    gm = _export(model, [4])

    c1 = torch.randn(4)
    c2 = c1 + 0.2 * torch.randn(4)

    shared_eps = expr.LpEpsilon([4])
    x = expr.Add(expr.ConstVal(c1), shared_eps)
    y = expr.Add(expr.ConstVal(c2), shared_eps)
    d = x - y

    op_diff = diff_interpret(gm)
    out = op_diff(DiffExpr3(x, y, d))
    diff_width = out.diff.ub() - out.diff.lb()

    op_std = zono.interpret(gm)
    y1 = op_std(expr.Add(expr.ConstVal(c1), expr.LpEpsilon([4])))
    y2 = op_std(expr.Add(expr.ConstVal(c2), expr.LpEpsilon([4])))
    naive_width = (y1.ub() - y2.lb()) - (y1.lb() - y2.ub())

    assert (diff_width < naive_width + 1e-5).all()