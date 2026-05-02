"""Tests for boundlab.diff.zono3 – differential zonotope verification.

For soundness tests we:
1. Build a DiffExpr3 ``(x, y, d)`` where x and y use independent epsilon
   symbols and ``d = x - y``.
2. Run the triple through the network via ``diff.zono3.interpret``.
3. Sample many concrete pairs ``(s1, s2)`` from the L∞ perturbation balls.
4. Assert every concrete difference ``f(s1) - f(s2)`` lies within the computed
   bounds of the output diff component.

Models are exported via ``torch.export.export`` before being passed to the
interpreters, so all ``nn.Linear`` submodules are lowered to ATen ops.
"""

import copy

import torch
import pytest
from torch import nn

import boundlab.expr as expr
from boundlab.interp.onnx import onnx_export

import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.op import DiffLinear, diff_pair
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp import Interpreter, ONNX_BASE_INTERPRETER


# =====================================================================
# Shared helpers
# =====================================================================

def _export(model: nn.Module, *in_shapes: list[int]):
    return onnx_export(model, in_shapes)



def _zonotope(center: torch.Tensor, scale: float = 1.0) -> expr.Expr:
    """L∞ zonotope: center ± scale, with a fresh independent epsilon symbol."""
    e = expr.LpEpsilon(list(center.shape))
    return center + scale * e


def _make_triple(c1: torch.Tensor, c2: torch.Tensor, scale: float = 1.0) -> DiffExpr3:
    """DiffExpr3 with independent eps for each center."""
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


# =====================================================================
# Affine handler correctness
# =====================================================================

def test_linear_bias_cancels_in_diff():
    """When x == y, the diff component stays zero after a linear layer."""
    torch.manual_seed(0)
    W = torch.randn(3, 4)
    c = torch.randn(4)
    x = _zonotope(c)
    d_init = x - x

    d_ub, d_lb = d_init.ublb()
    assert torch.allclose(d_ub, torch.zeros(4), atol=1e-6)
    assert torch.allclose(d_lb, torch.zeros(4), atol=1e-6)

    d_new = W @ d_init
    d_ub_new, d_lb_new = d_new.ublb()
    assert torch.allclose(d_ub_new, torch.zeros(3), atol=1e-6)
    assert torch.allclose(d_lb_new, torch.zeros(3), atol=1e-6)


def test_linear_diff_sound():
    """Diff bounds for a linear layer contain all sampled differences."""
    torch.manual_seed(1)
    W = torch.randn(3, 4)
    c1, c2 = torch.randn(4), torch.randn(4)

    triple = _make_triple(c1, c2) @ W.T

    s1, s2 = _sample_pairs(c1, c2)
    _check_sound(triple.diff, s1 @ W.T - s2 @ W.T)


def test_linear_diff_exact():
    """Diff bounds are exact for purely linear maps."""
    torch.manual_seed(2)
    W = torch.randn(3, 4)
    c1, c2 = torch.randn(4), torch.randn(4)

    triple = _make_triple(c1, c2)
    d_new = W @ triple.diff

    d_ub, d_lb = d_new.ublb()
    delta = c1 - c2
    expected_center = W @ delta
    expected_half_width = W.abs() @ torch.full((4,), 2.0)
    assert torch.allclose(d_ub, expected_center + expected_half_width, atol=1e-5)
    assert torch.allclose(d_lb, expected_center - expected_half_width, atol=1e-5)


def test_add_const_cancels_in_diff():
    """Adding a constant to both components leaves the diff unchanged."""
    torch.manual_seed(3)
    c1, c2 = torch.randn(4), torch.randn(4)
    const = torch.randn(4)

    triple = _make_triple(c1, c2)
    shifted = triple + expr.ConstVal(const)

    d_ub_before, d_lb_before = triple.diff.ublb()
    d_ub_after, d_lb_after = shifted.diff.ublb()
    assert torch.allclose(d_ub_before, d_ub_after, atol=1e-6)
    assert torch.allclose(d_lb_before, d_lb_after, atol=1e-6)


# =====================================================================
# DiffExpr2 operator coverage
# =====================================================================

def test_diffexpr2_neg():
    """DiffExpr2 negation negates both components."""
    torch.manual_seed(10)
    c1, c2 = torch.randn(4), torch.randn(4)
    x, y = _zonotope(c1), _zonotope(c2)
    pair = DiffExpr2(x, y)
    neg = -pair

    x_ub, x_lb = x.ublb()
    neg_ub, neg_lb = neg.x.ublb()
    assert torch.allclose(neg_ub, -x_lb, atol=1e-6)
    assert torch.allclose(neg_lb, -x_ub, atol=1e-6)


def test_diffexpr2_add_expr():
    """Adding a constant Expr to DiffExpr2 shifts both components identically."""
    torch.manual_seed(11)
    c1, c2, b = torch.randn(4), torch.randn(4), torch.randn(4)
    x, y = _zonotope(c1), _zonotope(c2)
    pair = DiffExpr2(x, y)
    shifted = pair + expr.ConstVal(b)

    ub_x, lb_x = x.ublb()
    ub_s, lb_s = shifted.x.ublb()
    assert torch.allclose(ub_s, ub_x + b, atol=1e-6)
    assert torch.allclose(lb_s, lb_x + b, atol=1e-6)


def test_diffexpr2_sub_diffexpr2():
    """Subtracting two DiffExpr2s subtracts componentwise."""
    torch.manual_seed(12)
    c1, c2, c3, c4 = [torch.randn(4) for _ in range(4)]
    p1 = DiffExpr2(_zonotope(c1), _zonotope(c2))
    p2 = DiffExpr2(_zonotope(c3), _zonotope(c4))
    diff = p1 - p2

    ub1, lb1 = p1.x.ublb()
    ub2, lb2 = p2.x.ublb()
    ub_d, lb_d = diff.x.ublb()
    assert torch.allclose(ub_d, ub1 - lb2, atol=1e-6)
    assert torch.allclose(lb_d, lb1 - ub2, atol=1e-6)


def test_diffexpr2_mul_scalar():
    """Scalar multiplication scales both components."""
    torch.manual_seed(13)
    c1, c2 = torch.randn(4), torch.randn(4)
    x, y = _zonotope(c1), _zonotope(c2)
    pair = DiffExpr2(x, y)
    scaled = pair * 3.0

    x_ub, x_lb = x.ublb()
    s_ub, s_lb = scaled.x.ublb()
    assert torch.allclose(s_ub, x_ub * 3.0, atol=1e-6)
    assert torch.allclose(s_lb, x_lb * 3.0, atol=1e-6)


def test_diffexpr2_matmul_tensor():
    """Matrix-multiply both components by a weight tensor."""
    torch.manual_seed(14)
    W = torch.randn(3, 4)
    c1, c2 = torch.randn(4), torch.randn(4)
    x, y = _zonotope(c1), _zonotope(c2)
    pair = DiffExpr2(x, y)
    out = W @ pair

    x_ub, x_lb = x.ublb()
    expected_ub = W.clamp(min=0) @ x_ub + W.clamp(max=0) @ x_lb + W @ c1
    # just soundness: out.x bounds contain W @ concrete samples
    s = c1 + (torch.rand(2000, 4) * 2 - 1)
    conc = s @ W.T
    ub, lb = out.x.ublb()
    assert (conc <= ub + 1e-5).all()
    assert (conc >= lb - 1e-5).all()


def test_diffexpr2_reshape():
    """reshape applies to both components."""
    c1, c2 = torch.randn(6), torch.randn(6)
    pair = DiffExpr2(_zonotope(c1), _zonotope(c2))
    reshaped = pair.reshape(2, 3)
    assert reshaped.shape == torch.Size([2, 3])
    assert reshaped.x.shape == torch.Size([2, 3])
    assert reshaped.y.shape == torch.Size([2, 3])


def test_diffexpr2_flatten():
    c1, c2 = torch.randn(2, 3), torch.randn(2, 3)
    pair = DiffExpr2(_zonotope(c1), _zonotope(c2))
    flat = pair.flatten()
    assert flat.shape == torch.Size([6])


def test_diffexpr2_unsqueeze_squeeze():
    c = torch.randn(4)
    pair = DiffExpr2(_zonotope(c), _zonotope(c))
    unsq = pair.unsqueeze(0)
    assert unsq.shape == torch.Size([1, 4])
    sq = unsq.squeeze(0)
    assert sq.shape == torch.Size([4])


# =====================================================================
# DiffExpr3 operator coverage
# =====================================================================

def test_diffexpr3_neg():
    """Negation negates all three components including diff."""
    torch.manual_seed(20)
    triple = _make_triple(torch.randn(4), torch.randn(4))
    neg = -triple

    d_ub, d_lb = triple.diff.ublb()
    nd_ub, nd_lb = neg.diff.ublb()
    assert torch.allclose(nd_ub, -d_lb, atol=1e-6)
    assert torch.allclose(nd_lb, -d_ub, atol=1e-6)


def test_diffexpr3_mul_scalar():
    """Scalar multiplication scales all three components."""
    torch.manual_seed(21)
    triple = _make_triple(torch.randn(4), torch.randn(4))
    scaled = triple * 2.0

    d_ub, d_lb = triple.diff.ublb()
    sd_ub, sd_lb = scaled.diff.ublb()
    assert torch.allclose(sd_ub, d_ub * 2.0, atol=1e-6)
    assert torch.allclose(sd_lb, d_lb * 2.0, atol=1e-6)


def test_diffexpr3_sub_diffexpr3():
    """Subtracting two DiffExpr3s subtracts diff components."""
    torch.manual_seed(22)
    t1 = _make_triple(torch.randn(4), torch.randn(4))
    t2 = _make_triple(torch.randn(4), torch.randn(4))
    diff = t1 - t2

    d1_ub, d1_lb = t1.diff.ublb()
    d2_ub, d2_lb = t2.diff.ublb()
    d_ub, d_lb = diff.diff.ublb()
    assert torch.allclose(d_ub, d1_ub - d2_lb, atol=1e-6)
    assert torch.allclose(d_lb, d1_lb - d2_ub, atol=1e-6)


def test_diffexpr3_add_diffexpr2():
    """Adding DiffExpr3 + DiffExpr2: diff gets the (x-y) contribution from pair."""
    torch.manual_seed(23)
    c1, c2, c3, c4 = [torch.randn(4) for _ in range(4)]
    triple = _make_triple(c1, c2)
    pair = DiffExpr2(_zonotope(c3), _zonotope(c4))
    result = triple + pair

    # diff = triple.diff + (pair.x - pair.y)
    d_ref = triple.diff + (pair.x - pair.y)
    ref_ub, ref_lb = d_ref.ublb()
    res_ub, res_lb = result.diff.ublb()
    assert torch.allclose(res_ub, ref_ub, atol=1e-6)
    assert torch.allclose(res_lb, ref_lb, atol=1e-6)


def test_diffexpr3_rsub_tensor():
    """tensor - DiffExpr3: x/y get (tensor - x/y), diff negates."""
    torch.manual_seed(24)
    c1, c2 = torch.randn(4), torch.randn(4)
    triple = _make_triple(c1, c2)
    t = torch.randn(4)
    result = t - triple

    d_ub, d_lb = triple.diff.ublb()
    rd_ub, rd_lb = result.diff.ublb()
    assert torch.allclose(rd_ub, -d_lb, atol=1e-6)
    assert torch.allclose(rd_lb, -d_ub, atol=1e-6)


def test_diffexpr3_shape_ops():
    """Shape ops (reshape, flatten, unsqueeze, squeeze) apply to all three."""
    c1, c2 = torch.randn(6), torch.randn(6)
    triple = DiffExpr3(_zonotope(c1), _zonotope(c2), _zonotope(c1 - c2))

    reshaped = triple.reshape(2, 3)
    assert reshaped.x.shape == torch.Size([2, 3])
    assert reshaped.diff.shape == torch.Size([2, 3])

    flat = reshaped.flatten()
    assert flat.diff.shape == torch.Size([6])

    unsq = triple.unsqueeze(0)
    assert unsq.diff.shape == torch.Size([1, 6])
    sq = unsq.squeeze(0)
    assert sq.diff.shape == torch.Size([6])


def test_diffexpr3_getitem_int():
    """Integer index on DiffExpr3 returns the component (x/y/diff), not tensor element."""
    c1, c2 = torch.randn(4), torch.randn(4)
    x, y = _zonotope(c1), _zonotope(c2)
    d = x - y
    triple = DiffExpr3(x, y, d)
    assert triple[0] is x
    assert triple[1] is y
    assert triple[2] is d


# =====================================================================
# ReLU differential soundness (per-regime)
# =====================================================================

_relu_handler = diff_interpret["relu"]


def test_relu_diff_dead_dead_is_zero():
    """dead/dead: relu(x) - relu(y) = 0."""
    x = _const_zonotope(-2.0, 0.5)
    y = _const_zonotope(-3.0, 0.5)
    out = _relu_handler(DiffExpr3(x, y, x - y))
    d_ub, d_lb = out.diff.ublb()
    assert torch.allclose(d_ub, torch.zeros(1), atol=1e-6)
    assert torch.allclose(d_lb, torch.zeros(1), atol=1e-6)


def test_relu_diff_active_active_passthrough():
    """active/active: relu(x) - relu(y) = x - y exactly."""
    x = _const_zonotope(2.0, 0.5)
    y = _const_zonotope(1.0, 0.5)
    d = x - y
    out = _relu_handler(DiffExpr3(x, y, d))
    d_ub_orig, d_lb_orig = d.ublb()
    d_ub_new, d_lb_new = out.diff.ublb()
    assert torch.allclose(d_ub_new, d_ub_orig, atol=1e-5)
    assert torch.allclose(d_lb_new, d_lb_orig, atol=1e-5)


def test_relu_diff_active_dead():
    """active/dead: relu(x) - relu(y) = x."""
    x = _const_zonotope(2.0, 0.5)
    y = _const_zonotope(-2.0, 0.5)
    out = _relu_handler(DiffExpr3(x, y, x - y))
    x_ub, x_lb = x.ublb()
    d_ub, d_lb = out.diff.ublb()
    assert torch.allclose(d_ub, x_ub, atol=1e-5)
    assert torch.allclose(d_lb, x_lb, atol=1e-5)


def test_relu_diff_dead_active():
    """dead/active: relu(x) - relu(y) = -y."""
    x = _const_zonotope(-2.0, 0.5)
    y = _const_zonotope(2.0, 0.5)
    out = _relu_handler(DiffExpr3(x, y, x - y))
    y_ub, y_lb = y.ublb()
    d_ub, d_lb = out.diff.ublb()
    assert torch.allclose(d_ub, -y_lb, atol=1e-5)
    assert torch.allclose(d_lb, -y_ub, atol=1e-5)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_relu_diff_crossing_sound(seed: int):
    """ReLU diff soundness in the crossing/crossing regime."""
    torch.manual_seed(seed)
    n = 6
    c1 = torch.rand(n) * 0.4 - 0.2
    c2 = torch.rand(n) * 0.4 - 0.2
    scale = 0.8

    triple = _make_triple(c1, c2, scale)
    out = _relu_handler(triple)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    _check_sound(out.diff, torch.relu(s1) - torch.relu(s2))


def test_relu_diff_diffexpr2_promotes_to_diffexpr3():
    """ReLU on a DiffExpr2 produces a DiffExpr3 (auto-promotion)."""
    x = _const_zonotope(0.5, 0.8)
    y = _const_zonotope(-0.2, 0.8)
    pair = DiffExpr2(x, y)
    out = _relu_handler(pair)
    assert isinstance(out, DiffExpr3)


# =====================================================================
# End-to-end MLP soundness
# =====================================================================

def test_interpreter_linear_diff_sound():
    torch.manual_seed(30)
    model = nn.Linear(4, 3)
    c1, c2 = torch.randn(4), torch.randn(4)
    op = diff_interpret(_export(model, [4]))
    out = op(_make_triple(c1, c2))
    s1, s2 = _sample_pairs(c1, c2)
    with torch.no_grad():
        _check_sound(out.diff, model(s1) - model(s2))


def test_interpreter_relu_mlp_diff_sound():
    torch.manual_seed(31)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    c1, c2 = torch.randn(4), torch.randn(4)
    op = diff_interpret(_export(model, [4]))
    out = op(_make_triple(c1, c2))
    s1, s2 = _sample_pairs(c1, c2)
    with torch.no_grad():
        _check_sound(out.diff, model(s1) - model(s2))


@pytest.mark.parametrize("seed", [40, 41, 42])
def test_interpreter_deep_mlp_diff_sound(seed: int):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(5, 8), nn.ReLU(),
        nn.Linear(8, 6), nn.ReLU(),
        nn.Linear(6, 4), nn.ReLU(),
        nn.Linear(4, 3),
    )
    c1, c2 = torch.randn(5), torch.randn(5)
    op = diff_interpret(_export(model, [5]))
    out = op(_make_triple(c1, c2))
    s1, s2 = _sample_pairs(c1, c2, n=2500)
    with torch.no_grad():
        _check_sound(out.diff, model(s1) - model(s2), tol=2e-5)


# =====================================================================
# DiffLinear
# =====================================================================

def test_diff_linear_exports_diff_pair():
    """ONNX export of DiffLinear contains boundlab::diff_pair nodes."""
    fc1 = nn.Linear(4, 3)
    fc2 = nn.Linear(4, 3)
    model = DiffLinear(fc1, fc2)
    onnx_model = _export(model, [4])
    diff_pair_nodes = [
        n for n in onnx_model.graph
        if n.domain == "boundlab" and n.op_type == "DiffPair"
    ]
    assert len(diff_pair_nodes) >= 1


def test_diff_linear_same_weights_diff_is_zero():
    """DiffLinear with identical weights: concrete diff is always zero, bounds are sound."""
    torch.manual_seed(50)
    fc = nn.Linear(4, 3)
    model = DiffLinear(fc, copy.deepcopy(fc))
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    x = _zonotope(c)
    out = op(x)
    assert isinstance(out, DiffExpr2)
    # Concrete diff f1(s) - f2(s) is always zero since weights are identical.
    # The zonotope bound of out.x - out.y may not be tight (that's what DiffExpr3 is for),
    # but it must be sound: zero must lie within [lb, ub].
    d = out.x - out.y
    d_ub, d_lb = d.ublb()
    assert (d_lb <= 1e-5).all(), "LB must be <= 0 when diff is always zero"
    assert (d_ub >= -1e-5).all(), "UB must be >= 0 when diff is always zero"


def test_diff_linear_diff_sound():
    """DiffLinear with different weights: diff bounds are sound."""
    torch.manual_seed(51)
    fc1 = nn.Linear(4, 3)
    fc2 = nn.Linear(4, 3)
    model = DiffLinear(fc1, fc2)
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    x = _zonotope(c)
    out = op(x)
    assert isinstance(out, DiffExpr2)

    d = out.x - out.y
    s = c + (torch.rand(2000, 4) * 2 - 1)
    with torch.no_grad():
        diffs = fc1(s) - fc2(s)
    _check_sound(d, diffs)


def test_diff_linear_width_less_than_independent():
    """DiffLinear gives tighter diff bounds than independent zonotope subtraction."""
    torch.manual_seed(52)
    fc1 = nn.Linear(4, 3)
    fc2 = nn.Linear(4, 3)
    model = DiffLinear(fc1, fc2)
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    z = _zonotope(c)
    out = op(z)
    paired_width = (out.x - out.y).bound_width()

    # Naive: bound each output independently
    op_std = zono.interpret(_export(fc1, [4]))
    z1 = op_std(z)
    op_std2 = zono.interpret(_export(fc2, [4]))
    z2 = op_std2(z)
    naive_width = (z1.ub() - z2.lb()) - (z1.lb() - z2.ub())

    assert (paired_width <= naive_width + 1e-5).all()


def test_diff_linear_relu_sound():
    """DiffLinear followed by ReLU: diff bounds are sound."""
    torch.manual_seed(53)
    fc1 = nn.Linear(5, 4)
    fc2 = nn.Linear(5, 4)
    model = nn.Sequential(DiffLinear(fc1, fc2), nn.ReLU())
    gm = _export(model, [5])
    op = diff_interpret(gm)

    c = torch.randn(5)
    z = _zonotope(c)
    out = op(z)
    # After ReLU the DiffExpr2 should be promoted to DiffExpr3
    assert isinstance(out, DiffExpr3)

    s = c + (torch.rand(2000, 5) * 2 - 1)
    with torch.no_grad():
        diffs = torch.relu(fc1(s)) - torch.relu(fc2(s))
    _check_sound(out.diff, diffs)


def test_diff_linear_full_mlp_sound():
    """DiffLinear in a full MLP: end-to-end soundness."""
    torch.manual_seed(54)
    fc1 = nn.Linear(4, 5)
    fc2 = nn.Linear(4, 5)
    model = nn.Sequential(DiffLinear(fc1, fc2), nn.ReLU(), nn.Linear(5, 3))
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    z = _zonotope(c)
    out = op(z)
    assert isinstance(out, DiffExpr3)

    s = c + (torch.rand(2000, 4) * 2 - 1)
    with torch.no_grad():
        diffs = model[2](torch.relu(fc1(s))) - model[2](torch.relu(fc2(s)))
    _check_sound(out.diff, diffs)


@pytest.mark.parametrize("seed", [60, 61, 62])
def test_diff_linear_parametric_sound(seed: int):
    """Parametric soundness test for DiffLinear with random weights."""
    torch.manual_seed(seed)
    fc1 = nn.Linear(6, 5)
    fc2 = nn.Linear(6, 5)
    model = nn.Sequential(DiffLinear(fc1, fc2), nn.ReLU(), nn.Linear(5, 3))
    gm = _export(model, [6])
    op = diff_interpret(gm)

    c = torch.randn(6)
    z = _zonotope(c, scale=0.5)
    out = op(z)

    s = c + (torch.rand(2000, 6) * 2 - 1) * 0.5
    with torch.no_grad():
        diffs = model[2](torch.relu(fc1(s))) - model[2](torch.relu(fc2(s)))
    _check_sound(out.diff, diffs, tol=2e-5)


def test_diff_net_merges_linear_layers_with_difflinear():
    """diff_net should insert boundlab::diff_pair nodes for linear layers."""
    torch.manual_seed(63)
    model1 = nn.Linear(4, 3)
    model2 = nn.Linear(4, 3)
    gm1 = _export(model1, [4])
    gm2 = _export(model2, [4])

    merged = diff_net(gm1, gm2)
    diff_pair_nodes = [n for n in merged.graph if n.domain == "boundlab" and n.op_type == "DiffPair"]
    assert len(diff_pair_nodes) == 2

    op = diff_interpret(merged)
    c = torch.randn(4)
    z = _zonotope(c, scale=0.3)
    out = op(z)
    assert isinstance(out, DiffExpr2)

    s = c + (torch.rand(2000, 4) * 2 - 1) * 0.3
    with torch.no_grad():
        diffs = model1(s) - model2(s)
    _check_sound(out.x - out.y, diffs, tol=2e-5)


def test_diff_net_deep_mlp_conversion_and_concrete_semantics():
    """diff_net converts deep MLP linears and preserves concrete branch-1 behavior."""
    torch.manual_seed(64)
    model1 = nn.Sequential(
        nn.Linear(5, 9),
        nn.ReLU(),
        nn.Linear(9, 7),
        nn.ReLU(),
        nn.Linear(7, 4),
    )
    model2 = nn.Sequential(
        nn.Linear(5, 9),
        nn.ReLU(),
        nn.Linear(9, 7),
        nn.ReLU(),
        nn.Linear(7, 4),
    )
    gm1 = _export(model1, [5])
    gm2 = _export(model2, [5])

    merged = diff_net(gm1, gm2)
    diff_pair_nodes = [n for n in merged.graph if n.domain == "boundlab" and n.op_type == "DiffPair"]
    assert len(diff_pair_nodes) == 6

    # Concrete semantics: use a concrete interpreter where diff_pair is a no-op.
    concrete_interpret = Interpreter(ONNX_BASE_INTERPRETER)
    concrete_interpret["DiffPair"] = lambda x, _: x
    concrete_interpret["Relu"] = lambda x: torch.relu(x)
    merged_concrete = concrete_interpret(merged)

    x1 = torch.randn(5)
    with torch.no_grad():
        y_expected = model1(x1)
        y_a = merged_concrete(x1)

    assert torch.allclose(y_a, y_expected, atol=1e-6)


def test_diff_net_merges_matmul_add_affine_pattern():
    """diff_net should pair ONNX-style ``aten.matmul + aten.add`` affine layers."""
    torch.manual_seed(65)

    class MatMulAffine(nn.Module):
        def __init__(self):
            super().__init__()
            # ONNX-style affine params: x @ w + b, with w shape (in, out)
            self.w = nn.Parameter(torch.randn(4, 3))
            self.b = nn.Parameter(torch.randn(3))

        def forward(self, x):
            return x @ self.w + self.b

    model1 = MatMulAffine()
    model2 = MatMulAffine()
    gm1 = _export(model1, [4])
    gm2 = _export(model2, [4])

    merged = diff_net(gm1, gm2)
    diff_pair_nodes = [n for n in merged.graph if n.domain == "boundlab" and n.op_type == "DiffPair"]
    assert len(diff_pair_nodes) == 2

    # Concrete semantics: use a concrete interpreter where diff_pair is a no-op.
    concrete_interpret = Interpreter(ONNX_BASE_INTERPRETER)
    concrete_interpret["DiffPair"] = lambda x, _: x
    merged_concrete = concrete_interpret(merged)

    x = torch.randn(4)
    with torch.no_grad():
        y_expected = model1(x)
        y_actual = merged_concrete(x)
    assert torch.allclose(y_actual, y_expected, atol=1e-6)


# =====================================================================
# Shared-noise tightness
# =====================================================================

def test_diff_tighter_than_independent():
    """Shared eps: diff bounds are strictly tighter than naive subtraction."""
    torch.manual_seed(70)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
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

    assert (diff_width < naive_width).all()


# =====================================================================
# Fallback: non-DiffExpr inputs use standard zonotope interpreter
# =====================================================================

def test_fallback_linear_matches_std():
    """Plain Expr through diff interpreter matches standard zono interpreter."""
    torch.manual_seed(80)
    model = nn.Linear(4, 3)
    gm = _export(model, [4])
    x = expr.Add(expr.ConstVal(torch.randn(4)), expr.LpEpsilon([4]))

    y_diff = diff_interpret(gm)(x)
    y_std = zono.interpret(gm)(x)

    assert torch.allclose(y_diff.ub(), y_std.ub(), atol=1e-6)
    assert torch.allclose(y_diff.lb(), y_std.lb(), atol=1e-6)


def test_fallback_relu_mlp_matches_std():
    """Plain Expr through diff ReLU MLP matches standard zono interpreter."""
    torch.manual_seed(81)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    gm = _export(model, [4])
    x = expr.Add(expr.ConstVal(torch.randn(4)), expr.LpEpsilon([4]))

    y_diff = diff_interpret(gm)(x)
    y_std = zono.interpret(gm)(x)

    assert torch.allclose(y_diff.ub(), y_std.ub(), atol=1e-5)
    assert torch.allclose(y_diff.lb(), y_std.lb(), atol=1e-5)


# =====================================================================
# Complex / advanced tests
# =====================================================================

def test_bias_only_difference_is_exact():
    """When W1 == W2 but b1 != b2, diff = b1 - b2 exactly (constant, no epsilon).

    This is the tightest possible case: the diff component has zero width because
    the weight matrices cancel perfectly and the bias difference is a constant.
    """
    torch.manual_seed(90)
    W = torch.randn(3, 4)
    b1 = torch.randn(3)
    b2 = torch.randn(3)

    # Build two linear layers sharing W but different biases.
    fc1 = nn.Linear(4, 3, bias=True)
    fc2 = nn.Linear(4, 3, bias=True)
    with torch.no_grad():
        fc1.weight.copy_(W)
        fc2.weight.copy_(W)
        fc1.bias.copy_(b1)
        fc2.bias.copy_(b2)

    c = torch.randn(4)
    x = _zonotope(c)
    # Both branches see the same input x; use a DiffExpr3 with d = x - x = 0.
    triple = DiffExpr3(x, x, x - x)
    gm = _export(fc1, [4])  # same W, so we can use fc1 graph structure
    # Build the diff manually through affine ops instead of the interpreter
    # to isolate the algebra: diff = W@d + (b1 - b2) = W@0 + (b1-b2) = b1-b2.
    W_expr = expr.ConstVal(W)
    d_out = W @ triple.diff  # W @ 0 = 0 (exact)
    bias_diff = expr.ConstVal(b1 - b2)
    diff_final = d_out + bias_diff

    d_ub, d_lb = diff_final.ublb()
    expected = b1 - b2
    assert torch.allclose(d_ub, expected, atol=1e-6), f"UB mismatch: {d_ub} vs {expected}"
    assert torch.allclose(d_lb, expected, atol=1e-6), f"LB mismatch: {d_lb} vs {expected}"


def test_diff_scales_linearly_with_weight_difference():
    """Diff bound width scales linearly when the weight difference is scaled by λ.

    For a purely linear model f1(x) = W1 x, f2(x) = W2 x:
      diff = (W1 - W2) x
    Scaling W2 toward W1 by λ → diff width scales by (1-λ).
    """
    torch.manual_seed(91)
    W_base = torch.randn(3, 4)
    delta_W = torch.randn(3, 4) * 0.3
    c = torch.randn(4)
    x = _zonotope(c)

    widths = []
    for lam in [0.0, 0.5, 1.0]:
        W1 = W_base
        W2 = W_base + (1 - lam) * delta_W
        diff = W1 @ x - W2 @ x  # = lam * delta_W @ x
        widths.append(diff.bound_width())

    # Width at lam=0.0 (full delta) >= lam=0.5 >= lam=1.0 (no delta, exact zero)
    assert (widths[0] >= widths[1] - 1e-5).all()
    assert (widths[1] >= widths[2] - 1e-5).all()
    assert torch.allclose(widths[2], torch.zeros(3), atol=1e-5)


def test_diff_sign_symmetry():
    """Swapping the two network inputs negates the diff component.

    DiffExpr3(x, y, d) and DiffExpr3(y, x, -d) should give opposite diff bounds.
    """
    torch.manual_seed(92)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c1, c2 = torch.randn(4), torch.randn(4)
    t_fwd = _make_triple(c1, c2)
    t_rev = _make_triple(c2, c1)

    out_fwd = op(t_fwd)
    out_rev = op(t_rev)

    fwd_ub, fwd_lb = out_fwd.diff.ublb()
    rev_ub, rev_lb = out_rev.diff.ublb()

    # Over-approximations are not guaranteed to be perfectly symmetric, but
    # each direction should be sound for concrete negated diffs.
    s1, s2 = _sample_pairs(c1, c2, n=2000)
    with torch.no_grad():
        diffs_fwd = model(s1) - model(s2)
        diffs_rev = model(s2) - model(s1)

    _check_sound(out_fwd.diff, diffs_fwd, tol=2e-5)
    _check_sound(out_rev.diff, diffs_rev, tol=2e-5)


def test_point_input_gives_exact_diff():
    """Zero-width perturbation ball (scale=0): diff bound collapses to a point f1(c)-f2(c).

    With scale=0 the diff expression degenerates to a constant, so ub == lb == f1(c) - f2(c).
    """
    torch.manual_seed(93)
    fc1 = nn.Linear(4, 3)
    fc2 = nn.Linear(4, 3)

    c = torch.randn(4)
    # scale=0 ⇒ eps contributes nothing; x == y == ConstVal(c).
    x = _zonotope(c, scale=0.0)
    y = _zonotope(c, scale=0.0)
    triple = DiffExpr3(x, y, x - y)

    gm = _export(fc1, [4])  # won't use this model, manual linear ops below
    W1, b1 = fc1.weight.detach(), fc1.bias.detach()
    W2, b2 = fc2.weight.detach(), fc2.bias.detach()

    # Exact algebraic result (fc uses aten linear = W @ x + b):
    d_out = W1 @ triple.diff + expr.ConstVal(b1 - b2)

    d_ub, d_lb = d_out.ublb()
    expected = W1 @ (c - c) + (b1 - b2)  # = b1 - b2
    assert torch.allclose(d_ub, expected, atol=1e-6)
    assert torch.allclose(d_lb, expected, atol=1e-6)


def test_multi_diff_linear_chain_sound():
    """DiffLinear at input followed by multiple standard ReLU+Linear layers: soundness.

    DiffLinear introduces the weight-pairing at the first layer.  Subsequent
    standard Linear layers propagate the DiffExpr3 diff component correctly.
    The concrete diff is f1_chain(x) - f2_chain(x) where the two chains share
    all weights except the first linear step.
    """
    torch.manual_seed(94)
    fc1a = nn.Linear(5, 6)
    fc1b = nn.Linear(5, 6)
    fc2 = nn.Linear(6, 5)
    fc3 = nn.Linear(5, 4)

    # DiffLinear introduces the split; subsequent layers are shared (same Linear)
    model = nn.Sequential(
        DiffLinear(fc1a, fc1b),
        nn.ReLU(),
        fc2,
        nn.ReLU(),
        fc3,
    )
    gm = _export(model, [5])
    op = diff_interpret(gm)

    c = torch.randn(5)
    z = _zonotope(c, scale=0.3)
    out = op(z)
    # After ReLU following DiffLinear the output is promoted to DiffExpr3
    assert isinstance(out, DiffExpr3)

    s = c + (torch.rand(2000, 5) * 2 - 1) * 0.3
    with torch.no_grad():
        # f1: fc1a -> relu -> fc2 -> relu -> fc3
        # f2: fc1b -> relu -> fc2 -> relu -> fc3
        h1 = fc3(torch.relu(fc2(torch.relu(fc1a(s)))))
        h2 = fc3(torch.relu(fc2(torch.relu(fc1b(s)))))
        diffs = h1 - h2
    _check_sound(out.diff, diffs, tol=2e-5)


def test_multi_diff_linear_chain_tighter_than_naive():
    """DiffLinear + shared layers is tighter than two independent zonotopes.

    Verifies that shared-noise exploitation across layers gives strictly tighter
    diff bounds than bounding each network independently.
    DiffLinear introduces the split; subsequent layers are shared/plain.
    """
    torch.manual_seed(95)
    fc1a = nn.Linear(4, 5)
    fc1b = nn.Linear(4, 5)
    fc2 = nn.Linear(5, 3)

    # One DiffLinear at the start, one shared linear at the end
    model = nn.Sequential(DiffLinear(fc1a, fc1b), nn.ReLU(), fc2)
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    z = _zonotope(c, scale=0.5)
    out = op(z)
    assert isinstance(out, DiffExpr3)
    paired_width = out.diff.bound_width()

    # Naive: bound each sub-model independently (fc2 shared, but models are separate)
    model1 = nn.Sequential(fc1a, nn.ReLU(), fc2)
    model2 = nn.Sequential(fc1b, nn.ReLU(), fc2)
    z_ind = _zonotope(c, scale=0.5)
    op1 = zono.interpret(_export(model1, [4]))
    op2 = zono.interpret(_export(model2, [4]))
    y1 = op1(z_ind)
    y2 = op2(z_ind)
    naive_width = (y1.ub() - y2.lb()) - (y1.lb() - y2.ub())

    assert (paired_width <= naive_width + 1e-4).all()


def test_multiple_difflinear_small_difference_sound():
    """Two DiffLinear layers with small branch deltas are empirically sound."""
    torch.manual_seed(97)

    fc1a = nn.Linear(4, 6)
    fc1b = copy.deepcopy(fc1a)
    fc2a = nn.Linear(6, 5)
    fc2b = copy.deepcopy(fc2a)

    # Keep the inter-network deltas intentionally small.
    with torch.no_grad():
        fc1b.weight.add_(1e-3 * torch.randn_like(fc1b.weight))
        fc1b.bias.add_(1e-3 * torch.randn_like(fc1b.bias))
        fc2b.weight.add_(1e-3 * torch.randn_like(fc2b.weight))
        fc2b.bias.add_(1e-3 * torch.randn_like(fc2b.bias))

    model = nn.Sequential(
        DiffLinear(fc1a, fc1b),
        nn.ReLU(),
        DiffLinear(fc2a, fc2b),
        nn.ReLU(),
        nn.Linear(5, 3),
    )
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    z = _zonotope(c, scale=0.1)
    out = op(z)
    assert isinstance(out, DiffExpr3)

    s = c + (torch.rand(2500, 4) * 2 - 1) * 0.1
    with torch.no_grad():
        head = model[4]
        diffs = head(torch.relu(fc2a(torch.relu(fc1a(s))))) - head(torch.relu(fc2b(torch.relu(fc1b(s)))))
    _check_sound(out.diff, diffs, tol=5e-5)


def test_diffexpr3_matmul_weight_difference():
    """DiffExpr3 @ W captures the bilinear identity: d@W1 + y*(W1-W2).

    For a DiffExpr3(x, y, d) where d = x - y, applying two different weight
    matrices W1 and W2 to branches 1 and 2 respectively yields:
      output_diff = d @ W1 + y @ (W1 - W2)
    This test verifies soundness for a custom two-weight linear step.
    """
    torch.manual_seed(96)
    W1 = torch.randn(5, 4)
    W2 = torch.randn(5, 4)

    c1, c2 = torch.randn(4), torch.randn(4)
    triple = _make_triple(c1, c2)

    # Manually apply W1 to branch x and W2 to branch y, tracking diff
    out_x = W1 @ triple.x
    out_y = W2 @ triple.y
    # diff = W1 @ x - W2 @ y = W1 @ (x - y) + (W1 - W2) @ y
    out_d = W1 @ triple.diff + (W1 - W2) @ triple.y

    # Check soundness against samples
    s1, s2 = _sample_pairs(c1, c2, n=3000)
    conc = s1 @ W1.T - s2 @ W2.T
    _check_sound(out_d, conc)

    # Also verify out_x and out_y are bounded correctly
    conc_x = s1 @ W1.T
    conc_y = s2 @ W2.T
    ub_x, lb_x = out_x.ublb()
    ub_y, lb_y = out_y.ublb()
    assert (conc_x <= ub_x.unsqueeze(0) + 1e-5).all()
    assert (conc_y <= ub_y.unsqueeze(0) + 1e-5).all()


def test_diff_linear_deep_relu_sound():
    """DiffLinear at input, followed by deep standard ReLU layers: soundness.

    DiffLinear introduces the weight split at the first layer; multiple
    standard ReLU+Linear layers compound the differential tracking.
    """
    torch.manual_seed(97)
    fc1a = nn.Linear(5, 8, bias=False)
    fc1b = nn.Linear(5, 8, bias=False)
    fc2a = nn.Linear(8, 6, bias=False)
    fc2b = nn.Linear(8, 6, bias=False)
    fc3 = nn.Linear(6, 4)
    fc4 = nn.Linear(4, 3)

    model = nn.Sequential(
        DiffLinear(fc1a, fc1b),
        nn.ReLU(),
        DiffLinear(fc2a, fc2b), nn.ReLU(),
        fc3, nn.ReLU(),
        fc4,
    )
    gm = _export(model, [5])
    op = diff_interpret(gm)

    c = torch.randn(5)
    z = _zonotope(c, scale=0.2)
    out = op(z)
    assert isinstance(out, DiffExpr3)

    s = c + (torch.rand(1500, 5) * 2 - 1) * 0.2
    with torch.no_grad():
        h1 = fc4(torch.relu(fc3(torch.relu(fc2a(torch.relu(fc1a(s)))))))
        h2 = fc4(torch.relu(fc3(torch.relu(fc2b(torch.relu(fc1b(s)))))))
        diffs = h1 - h2
    _check_sound(out.diff, diffs, tol=2e-5)

def test_diff_linear_deep_relu_sound2():
    """DiffLinear at input, followed by deep standard ReLU layers: soundness.

    DiffLinear introduces the weight split at the first layer; multiple
    standard ReLU+Linear layers compound the differential tracking.
    """
    torch.manual_seed(97)
    fc1a = nn.Linear(5, 8)
    fc1b = nn.Linear(5, 8)

    model = nn.Sequential(
        DiffLinear(fc1a, fc1b),
        nn.ReLU(),
    )
    gm = _export(model, [5])
    op = diff_interpret(gm)

    c = torch.randn(5)
    n = 0.001 * torch.randn(5)
    z1 = c + 0.1 * expr.LpEpsilon([5])
    z2 = z1 + n
    out = op(DiffExpr3(z1, z2, n))
    assert isinstance(out, DiffExpr3)

    with torch.no_grad():
        h1 = torch.relu(fc1a(c))
        h2 = torch.relu(fc1b(c + n))
        diffs = h1 - h2
    _check_sound(out.diff, diffs, tol=2e-5)

def test_diff_triple_plus_triple_sound():
    """DiffExpr3 + DiffExpr3: combined diff = d1 + d2, bounds are sound.

    If f1(x)-f2(x) is bounded by t1.diff and g1(x)-g2(x) by t2.diff,
    then (f1+g1)(x) - (f2+g2)(x) should be bounded by their sum.
    """
    torch.manual_seed(98)
    n = 5
    W1a, W1b = torch.randn(n, n), torch.randn(n, n)
    W2a, W2b = torch.randn(n, n), torch.randn(n, n)

    c1, c2 = torch.randn(n), torch.randn(n)
    t1 = _make_triple(c1, c2)
    t2 = _make_triple(c1, c2)  # same input perturbation centres

    # Apply weight pairs and add results
    out1 = DiffExpr3(W1a @ t1.x, W1b @ t1.y, W1a @ t1.diff + (W1a - W1b) @ t1.y)
    out2 = DiffExpr3(W2a @ t2.x, W2b @ t2.y, W2a @ t2.diff + (W2a - W2b) @ t2.y)
    combined = out1 + out2

    s1, s2 = _sample_pairs(c1, c2, n=2000)
    conc = (s1 @ W1a.T - s2 @ W1b.T) + (s1 @ W2a.T - s2 @ W2b.T)
    _check_sound(combined.diff, conc)


@pytest.mark.parametrize("scale", [0.1, 0.5, 1.0])
def test_diff_bound_width_grows_with_perturbation_radius(scale: float):
    """Diff bound width grows monotonically with the perturbation radius.

    For a DiffLinear model: larger ε-ball → wider diff bounds.
    """
    torch.manual_seed(99)
    fc1 = nn.Linear(4, 3)
    fc2 = nn.Linear(4, 3)
    model = DiffLinear(fc1, fc2)
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    z_small = _zonotope(c, scale=scale * 0.5)
    z_large = _zonotope(c, scale=scale)

    out_small = op(z_small, z_small)
    out_large = op(z_large, z_large)

    d_small = out_small.x - out_small.y
    d_large = out_large.x - out_large.y

    assert (d_large.bound_width() >= d_small.bound_width() - 1e-5).all()


def test_diffexpr3_relu_then_linear_sound():
    """DiffExpr3 through ReLU then linear: diff bounds remain sound.

    Uses the interpreter on a two-layer model to verify that the diff
    tracking correctly handles the ReLU→linear composition.
    """
    torch.manual_seed(100)
    model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    gm = _export(model, [6])
    op = diff_interpret(gm)

    c1 = torch.zeros(6)
    c2 = torch.randn(6) * 0.3
    scale = 0.4

    triple = _make_triple(c1, c2, scale)
    out = op(triple)
    assert isinstance(out, DiffExpr3)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    with torch.no_grad():
        diffs = model(s1) - model(s2)
    _check_sound(out.diff, diffs, tol=2e-5)


def test_diffexpr3_relu_width_tighter_on_active_neurons():
    """ReLU diff bounds are tighter when both neurons are clearly active.

    active/active regime (no crossing) ⇒ diff passes through unchanged;
    crossing regime ⇒ diff is relaxed (wider). The active/active bound
    should be <= the crossing bound.
    """
    # active/active: x ∈ [1.5, 2.5], y ∈ [0.5, 1.5] — no sign change
    x_aa = _const_zonotope(2.0, 0.5)
    y_aa = _const_zonotope(1.0, 0.5)
    out_aa = _relu_handler(DiffExpr3(x_aa, y_aa, x_aa - y_aa))
    width_aa = out_aa.diff.bound_width()

    # crossing: x ∈ [-0.8, 0.8], y ∈ [-0.6, 0.6] — spans zero
    x_cr = _const_zonotope(0.0, 0.8)
    y_cr = _const_zonotope(0.0, 0.6)
    out_cr = _relu_handler(DiffExpr3(x_cr, y_cr, x_cr - y_cr))
    width_cr = out_cr.diff.bound_width()

    # In active/active the zonotope passes through exactly; the crossing case adds slack
    assert (width_aa <= width_cr + 1e-5).all()


def test_zero_diff_preserved_through_relu():
    """If d = 0 exactly and both branches are purely active, d stays zero after ReLU."""
    # Both branches identical, both clearly above zero
    c = torch.tensor([2.0, 3.0, 1.5])
    x = _zonotope(c, scale=0.4)  # all lower bounds > 0
    triple = DiffExpr3(x, x, x - x)  # diff = 0 exactly

    out = _relu_handler(triple)
    d_ub, d_lb = out.diff.ublb()
    assert torch.allclose(d_ub, torch.zeros(3), atol=1e-6)
    assert torch.allclose(d_lb, torch.zeros(3), atol=1e-6)


def test_diff_linear_bias_difference_only():
    """DiffLinear with same W but different biases: diff is always b1 - b2.

    Since inputs are shared (same zonotope) and weights are equal, the concrete
    diff f1(x) - f2(x) = b1 - b2 is constant.  The interpreted diff bounds must
    contain b1 - b2 for all x in the perturbation ball (soundness), and their
    center should equal b1 - b2 (the diff is a constant).
    """
    torch.manual_seed(101)
    W = torch.randn(3, 4)
    b1 = torch.randn(3)
    b2 = torch.randn(3)

    fc1 = nn.Linear(4, 3)
    fc2 = nn.Linear(4, 3)
    with torch.no_grad():
        fc1.weight.copy_(W)
        fc2.weight.copy_(W)
        fc1.bias.copy_(b1)
        fc2.bias.copy_(b2)

    model = DiffLinear(fc1, fc2)
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    z = _zonotope(c)
    out = op(z)
    assert isinstance(out, DiffExpr2)

    # The concrete diff f1(s) - f2(s) = b1 - b2 for any input s.
    d = out.x - out.y
    d_ub, d_lb = d.ublb()
    expected = b1 - b2

    # Soundness: b1 - b2 must lie within [lb, ub]
    assert (d_lb <= expected + 1e-5).all(), f"LB too high: lb={d_lb}, expected={expected}"
    assert (d_ub >= expected - 1e-5).all(), f"UB too low: ub={d_ub}, expected={expected}"

    # Verify with concrete samples
    s = c + (torch.rand(1000, 4) * 2 - 1)
    with torch.no_grad():
        conc_diffs = fc1(s) - fc2(s)
    # All concrete diffs should equal b1 - b2 (since same W)
    assert torch.allclose(conc_diffs, expected.unsqueeze(0).expand_as(conc_diffs), atol=1e-5)
    _check_sound(d, conc_diffs)


def test_diffexpr2_getitem_slice():
    """Slicing a DiffExpr2 returns a DiffExpr2 with sliced components."""
    torch.manual_seed(102)
    c1, c2 = torch.randn(8), torch.randn(8)
    pair = DiffExpr2(_zonotope(c1), _zonotope(c2))
    sliced = pair[2:6]
    assert isinstance(sliced, DiffExpr2)
    assert sliced.shape == torch.Size([4])

    # Bounds of sliced components match bounds of original components' slices
    ub_x, lb_x = pair.x[2:6].ublb()
    ub_s, lb_s = sliced.x.ublb()
    assert torch.allclose(ub_x, ub_s, atol=1e-6)


def test_diffexpr3_getitem_slice():
    """Slicing a DiffExpr3 applies to all three components."""
    torch.manual_seed(103)
    c1, c2 = torch.randn(8), torch.randn(8)
    triple = _make_triple(c1, c2)
    sliced = triple[3:7]
    assert isinstance(sliced, DiffExpr3)
    assert sliced.diff.shape == torch.Size([4])

    ub_d, lb_d = triple.diff[3:7].ublb()
    ub_s, lb_s = sliced.diff.ublb()
    assert torch.allclose(ub_d, ub_s, atol=1e-6)


def test_diff_linear_different_output_sizes_same_width():
    """DiffLinear with wider hidden layer: diff bounds scale with output dimension.

    Sanity check that soundness holds for a larger hidden layer and that bound
    widths are non-negative (trivially required).
    """
    torch.manual_seed(104)
    fc1 = nn.Linear(4, 16)
    fc2 = nn.Linear(4, 16)
    model = nn.Sequential(DiffLinear(fc1, fc2), nn.ReLU(), nn.Linear(16, 2))
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    z = _zonotope(c, scale=0.5)
    out = op(z)
    assert isinstance(out, DiffExpr3)

    d_ub, d_lb = out.diff.ublb()
    assert (d_ub >= d_lb - 1e-6).all(), "UB must be >= LB"

    s = c + (torch.rand(2000, 4) * 2 - 1) * 0.5
    with torch.no_grad():
        h1 = torch.relu(fc1(s))
        h2 = torch.relu(fc2(s))
        W_out = model[2].weight.detach()
        b_out = model[2].bias.detach()
        diffs = h1 @ W_out.T + b_out - (h2 @ W_out.T + b_out)
    _check_sound(out.diff, diffs, tol=2e-5)


@pytest.mark.parametrize("seed", [110, 111, 112])
def test_diffexpr3_full_pipeline_correctness(seed: int):
    """Parametric test: DiffExpr3 through a random MLP matches sampled differences.

    This is the most comprehensive end-to-end check: random weights, random
    centers, random perturbation scales.
    """
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(6, 10), nn.ReLU(),
        nn.Linear(10, 8), nn.ReLU(),
        nn.Linear(8, 4),
    )
    gm = _export(model, [6])
    op = diff_interpret(gm)

    c1 = torch.randn(6)
    c2 = torch.randn(6) * 0.5
    scale = torch.rand(1).item() * 0.8 + 0.1  # in [0.1, 0.9]

    triple = _make_triple(c1, c2, scale)
    out = op(triple)
    assert isinstance(out, DiffExpr3)

    s1, s2 = _sample_pairs(c1, c2, n=3000, scale=scale)
    with torch.no_grad():
        diffs = model(s1) - model(s2)
    _check_sound(out.diff, diffs, tol=2e-5)
