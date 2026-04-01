"""Tests for boundlab.diff.zono3 – differential zonotope verification.

For soundness tests we:
1. Build a DiffExpr3 ``(x, y, d)`` where x and y use independent epsilon
   symbols and ``d = x - y``.
2. Run the triple through the network via ``diff.zono3.interpret``.
3. Sample many concrete pairs ``(s1, s2)`` from the L∞ perturbation balls.
4. Assert every concrete difference ``f(s1) - f(s2)`` lies within the computed
   bounds of the output diff component.

Models are exported via ``torch.export.export`` before
being passed to the interpreters, so all ``nn.Linear`` submodules are lowered
to ``aten.linear.default`` call_function nodes.
"""

import torch
import pytest
from torch import nn

import boundlab.expr as expr
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.zono3 import interpret as diff_interpret


def _export(model: nn.Module, in_shape: list[int]):
    """Export *model* to a ``torch.export.ExportedProgram``."""
    return torch.export.export(model, (torch.zeros(in_shape),))


# =====================================================================
# Helpers
# =====================================================================

def _zonotope(center: torch.Tensor, scale: float = 1.0) -> expr.Expr:
    """L∞ zonotope: center ± scale, with a fresh independent eps symbol."""
    e = expr.LpEpsilon(list(center.shape))
    return expr.Add(expr.ConstVal(center), torch.full_like(center, scale) * e)


def _make_triple(c1: torch.Tensor, c2: torch.Tensor, scale: float = 1.0) -> DiffExpr3:
    """DiffExpr3 ``(x, y, x-y)`` with independent eps for each input."""
    x = _zonotope(c1, scale)
    y = _zonotope(c2, scale)
    return DiffExpr3(x, y, x - y)


def _sample_pairs(c1: torch.Tensor, c2: torch.Tensor, n: int = 2000, scale: float = 1.0):
    """Sample *n* concrete pairs from the L∞ ball of radius *scale*."""
    s1 = c1 + (torch.rand(n, *c1.shape) * 2 - 1) * scale
    s2 = c2 + (torch.rand(n, *c2.shape) * 2 - 1) * scale
    return s1, s2


def _check_d_sound(d_expr: expr.Expr, diffs: torch.Tensor, tol: float = 1e-5):
    """Assert all concrete diffs are contained in the d_expr bounds."""
    d_ub, d_lb = d_expr.ublb()
    assert (diffs <= d_ub.unsqueeze(0) + tol).all(), (
        f"Upper bound violated: max excess = {(diffs - d_ub.unsqueeze(0)).max():.6f}"
    )
    assert (diffs >= d_lb.unsqueeze(0) - tol).all(), (
        f"Lower bound violated: max deficit = {(d_lb.unsqueeze(0) - diffs).max():.6f}"
    )


# =====================================================================
# Affine handler correctness
# =====================================================================

def test_linear_bias_cancels_in_diff():
    """When x == y, the diff component stays zero after a linear layer."""
    torch.manual_seed(0)
    W = torch.randn(3, 4)
    b = torch.randn(3)
    c = torch.randn(4)

    # x == y → d starts at 0
    x = _zonotope(c)
    d_init = x - x  # same expr, bound propagation cancels exactly

    d_ub, d_lb = d_init.ublb()
    assert torch.allclose(d_ub, torch.zeros(4), atol=1e-6)
    assert torch.allclose(d_lb, torch.zeros(4), atol=1e-6)

    # After linear, d_new = W @ 0 + 0 (no bias) → still zero
    d_new = W @ d_init
    d_ub_new, d_lb_new = d_new.ublb()
    assert torch.allclose(d_ub_new, torch.zeros(3), atol=1e-6)
    assert torch.allclose(d_lb_new, torch.zeros(3), atol=1e-6)


def test_linear_diff_sound():
    """Diff bounds for a linear layer must contain all sampled differences."""
    torch.manual_seed(1)
    W = torch.randn(3, 4)
    b = torch.randn(3)
    c1, c2 = torch.randn(4), torch.randn(4)

    triple = _make_triple(c1, c2)
    x_new = W @ triple.x + expr.ConstVal(b)
    y_new = W @ triple.y + expr.ConstVal(b)
    d_new = W @ triple.diff  # bias cancels

    s1, s2 = _sample_pairs(c1, c2)
    diffs = s1 @ W.T - s2 @ W.T  # bias cancels in difference
    _check_d_sound(d_new, diffs)


def test_linear_diff_exact():
    """Diff bounds are exact for purely linear maps (no approximation)."""
    torch.manual_seed(2)
    W = torch.randn(3, 4)
    c1, c2 = torch.randn(4), torch.randn(4)

    triple = _make_triple(c1, c2)
    d_new = W @ triple.diff

    d_ub, d_lb = d_new.ublb()
    delta = c1 - c2
    expected_center = W @ delta
    # two independent unit balls → combined range is |W| @ 2·ones
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
# ReLU differential soundness (per-regime)
# =====================================================================

def _const_zonotope(center_val: float, half_width: float, n: int = 1):
    """1-D zonotope with given center and half-width."""
    c = torch.full((n,), center_val)
    e = expr.LpEpsilon([n])
    return expr.Add(expr.ConstVal(c), torch.full((n,), half_width) * e)


_relu_diff_handler = diff_interpret.dispatcher["relu"]


def test_relu_diff_dead_dead_is_zero():
    """Case 1 (dead, dead): relu(x) - relu(y) = 0."""
    x = _const_zonotope(-2.0, 0.5)   # x ∈ [-2.5, -1.5] → dead
    y = _const_zonotope(-3.0, 0.5)   # y ∈ [-3.5, -2.5] → dead
    d = x - y

    out = _relu_diff_handler(DiffExpr3(x, y, d))
    d_ub, d_lb = out.diff.ublb()
    assert torch.allclose(d_ub, torch.zeros(1), atol=1e-6)
    assert torch.allclose(d_lb, torch.zeros(1), atol=1e-6)


def test_relu_diff_active_active_passthrough():
    """Case 4 (active, active): relu(x) - relu(y) = x - y exactly."""
    x = _const_zonotope(2.0, 0.5)   # x ∈ [1.5, 2.5] → active
    y = _const_zonotope(1.0, 0.5)   # y ∈ [0.5, 1.5] → active
    d = x - y

    out = _relu_diff_handler(DiffExpr3(x, y, d))

    # d_new should equal d; check bounds match
    d_ub_orig, d_lb_orig = d.ublb()
    d_ub_new, d_lb_new = out.diff.ublb()
    assert torch.allclose(d_ub_new, d_ub_orig, atol=1e-5)
    assert torch.allclose(d_lb_new, d_lb_orig, atol=1e-5)


def test_relu_diff_active_dead():
    """Case 3 (active, dead): relu(x) - relu(y) = x."""
    x = _const_zonotope(2.0, 0.5)   # active
    y = _const_zonotope(-2.0, 0.5)  # dead
    d = x - y

    out = _relu_diff_handler(DiffExpr3(x, y, d))

    # d_new should have same bounds as x
    x_ub, x_lb = x.ublb()
    d_ub, d_lb = out.diff.ublb()
    assert torch.allclose(d_ub, x_ub, atol=1e-5)
    assert torch.allclose(d_lb, x_lb, atol=1e-5)


def test_relu_diff_dead_active():
    """Case 2 (dead, active): relu(x) - relu(y) = -y."""
    x = _const_zonotope(-2.0, 0.5)  # dead
    y = _const_zonotope(2.0, 0.5)   # active
    d = x - y

    out = _relu_diff_handler(DiffExpr3(x, y, d))

    # d_new should have same bounds as -y
    y_ub, y_lb = y.ublb()
    d_ub, d_lb = out.diff.ublb()
    assert torch.allclose(d_ub, -y_lb, atol=1e-5)
    assert torch.allclose(d_lb, -y_ub, atol=1e-5)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_relu_diff_crossing_sound(seed: int):
    """ReLU diff soundness when both x and y are in the crossing regime."""
    torch.manual_seed(seed)
    n = 6
    c1 = torch.rand(n) * 0.4 - 0.2   # small centers near 0
    c2 = torch.rand(n) * 0.4 - 0.2
    scale = 0.8  # ensures crossing for most neurons

    triple = _make_triple(c1, c2, scale)
    out = _relu_diff_handler(triple)

    s1, s2 = _sample_pairs(c1, c2, scale=scale, n=3000)
    diffs = torch.relu(s1) - torch.relu(s2)
    _check_d_sound(out.diff, diffs)


# =====================================================================
# End-to-end MLP soundness via interpret()
# =====================================================================

def test_interpreter_linear_diff_sound():
    """diff.zono3.interpret on nn.Linear produces sound diff bounds."""
    torch.manual_seed(20)
    model = nn.Linear(4, 3)
    c1, c2 = torch.randn(4), torch.randn(4)

    op = diff_interpret(_export(model, [4]))
    triple = _make_triple(c1, c2)
    out = op(triple)

    s1, s2 = _sample_pairs(c1, c2)
    with torch.no_grad():
        diffs = model(s1) - model(s2)
    _check_d_sound(out.diff, diffs)


def test_interpreter_relu_mlp_diff_sound():
    """diff.zono3.interpret on a ReLU MLP produces sound diff bounds."""
    torch.manual_seed(21)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    c1, c2 = torch.randn(4), torch.randn(4)

    op = diff_interpret(_export(model, [4]))
    triple = _make_triple(c1, c2)
    out = op(triple)

    s1, s2 = _sample_pairs(c1, c2)
    with torch.no_grad():
        diffs = model(s1) - model(s2)
    _check_d_sound(out.diff, diffs)


@pytest.mark.parametrize("seed", [30, 31, 32])
def test_interpreter_deep_mlp_diff_sound(seed: int):
    """Stress-test soundness on deeper ReLU MLPs."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(5, 8), nn.ReLU(),
        nn.Linear(8, 6), nn.ReLU(),
        nn.Linear(6, 4), nn.ReLU(),
        nn.Linear(4, 3),
    )
    c1, c2 = torch.randn(5), torch.randn(5)

    op = diff_interpret(_export(model, [5]))
    triple = _make_triple(c1, c2)
    out = op(triple)

    s1, s2 = _sample_pairs(c1, c2, n=2500)
    with torch.no_grad():
        diffs = model(s1) - model(s2)
    _check_d_sound(out.diff, diffs, tol=2e-5)


def test_diff_tighter_than_independent():
    """With a shared input perturbation, diff bounds are tighter than naive subtraction.

    When x and y share the same LpEpsilon (same perturbation structure) but
    have slightly different centers, the diff cancels the shared noise exactly.
    The naive approach of subtracting independent bound intervals is much looser.
    """
    torch.manual_seed(40)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    gm = _export(model, [4])

    c1 = torch.randn(4)
    c2 = c1 + 0.2 * torch.randn(4)  # nearby center

    # Shared eps: both x and y are perturbed by the same noise symbol
    shared_eps = expr.LpEpsilon([4])
    x = expr.Add(expr.ConstVal(c1), shared_eps)
    y = expr.Add(expr.ConstVal(c2), shared_eps)
    d = x - y  # = ConstVal(c1 - c2), shared noise cancels exactly

    op_diff = diff_interpret(gm)
    out = op_diff(DiffExpr3(x, y, d))
    diff_width = out.diff.ub() - out.diff.lb()

    # Naive: treat x and y as independent zonotopes, subtract bound intervals
    op_std = zono.interpret(gm)
    y1 = op_std(expr.Add(expr.ConstVal(c1), expr.LpEpsilon([4])))
    y2 = op_std(expr.Add(expr.ConstVal(c2), expr.LpEpsilon([4])))
    naive_width = (y1.ub() - y2.lb()) - (y1.lb() - y2.ub())

    assert (diff_width < naive_width).all(), (
        "Diff bounds with shared eps should be strictly tighter than naive subtraction"
    )


# =====================================================================
# Fallback: non-DiffExpr3 input uses standard interpreter
# =====================================================================

def test_fallback_linear_matches_std():
    """Plain Expr through diff interpreter matches standard zono interpreter."""
    torch.manual_seed(50)
    model = nn.Linear(4, 3)
    gm = _export(model, [4])
    c = torch.randn(4)
    e = expr.LpEpsilon([4])
    x = expr.Add(expr.ConstVal(c), e)

    y_diff = diff_interpret(gm)(x)
    y_std = zono.interpret(gm)(x)

    assert torch.allclose(y_diff.ub(), y_std.ub(), atol=1e-6)
    assert torch.allclose(y_diff.lb(), y_std.lb(), atol=1e-6)


def test_fallback_relu_mlp_matches_std():
    """Plain Expr through diff ReLU MLP matches standard zono interpreter."""
    torch.manual_seed(51)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    gm = _export(model, [4])
    c = torch.randn(4)
    e = expr.LpEpsilon([4])
    x = expr.Add(expr.ConstVal(c), e)

    y_diff = diff_interpret(gm)(x)
    y_std = zono.interpret(gm)(x)

    assert torch.allclose(y_diff.ub(), y_std.ub(), atol=1e-5)
    assert torch.allclose(y_diff.lb(), y_std.lb(), atol=1e-5)
