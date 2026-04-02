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
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.diff.op import DiffLinear, diff_pair
from boundlab.diff.zono3 import interpret as diff_interpret


# =====================================================================
# Shared helpers
# =====================================================================

def _export(model: nn.Module, *in_shapes: list[int]):
    args = tuple(torch.zeros(s) for s in in_shapes)
    return torch.export.export(model, args)


def _zonotope(center: torch.Tensor, scale: float = 1.0) -> expr.Expr:
    """L∞ zonotope: center ± scale, with a fresh independent epsilon symbol."""
    e = expr.LpEpsilon(list(center.shape))
    return expr.Add(expr.ConstVal(center), torch.full_like(center, scale) * e)


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

    triple = _make_triple(c1, c2)
    d_new = W @ triple.diff

    s1, s2 = _sample_pairs(c1, c2)
    _check_sound(d_new, s1 @ W.T - s2 @ W.T)


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

_relu_handler = diff_interpret.dispatcher["relu"]


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
    """Exported graph of DiffLinear contains a diff_pair node."""
    fc1 = nn.Linear(4, 3)
    fc2 = nn.Linear(4, 3)
    model = DiffLinear(fc1, fc2)
    gm = _export(model, [4])
    node_names = [n.target.__name__ for n in gm.graph.nodes if n.op == "call_function"]
    assert any("diff_pair" in name for name in node_names)


def test_diff_linear_same_weights_diff_is_zero():
    """DiffLinear with identical weights: concrete diff is always zero, bounds are sound."""
    torch.manual_seed(50)
    fc = nn.Linear(4, 3)
    model = DiffLinear(fc, copy.deepcopy(fc))
    gm = _export(model, [4])
    op = diff_interpret(gm)

    c = torch.randn(4)
    x = _zonotope(c)
    out = op(x, x)
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
    out = op(x, x)
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
    out = op(z, z)
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
    out = op(z, z)
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
    out = op(z, z)
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
    out = op(z, z)

    s = c + (torch.rand(2000, 6) * 2 - 1) * 0.5
    with torch.no_grad():
        diffs = model[2](torch.relu(fc1(s))) - model[2](torch.relu(fc2(s)))
    _check_sound(out.diff, diffs, tol=2e-5)


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
