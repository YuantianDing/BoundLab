"""Tests for boundlab.diff.zonohex - hexagonal zonotope relaxation for
differential verification.

The zonohex interpreter wraps :mod:`boundlab.diff.zono3` by inserting a
:class:`ZonoHexGate` (applied via :func:`expr3_to_expr2`) after every operator.
The gate takes the triple ``(x, y, diff)`` and produces a pair of bounded
outputs ``(x_bounded, y_bounded)`` whose concretised intervals must still
contain the true network outputs, while (optionally) exploiting the diff
component to tighten them.

These tests check:

1. Soundness: sampled ``(f1(s1), f2(s2))`` fall inside the per-network bounds
   for the gate outputs, and ``f1(s1) - f2(s2)`` falls inside the induced diff
   interval ``[lb_x - ub_y, ub_x - lb_y]``.
2. The interpreter falls back gracefully for non-DiffExpr3 inputs.
"""

import torch
import pytest
from torch import nn

import boundlab.expr as expr
from boundlab.diff.zono3.expr import DiffExpr2, DiffExpr3
from boundlab.diff.zonohex import interpret as hex_interpret, ZonoHexGate, expr3_to_expr2
from boundlab.interp.onnx import onnx_export


# =====================================================================
# Helpers
# =====================================================================

def _zonotope(center: torch.Tensor, scale: float = 1.0) -> expr.Expr:
    return center + scale * expr.LpEpsilon(list(center.shape))


def _make_triple(c1: torch.Tensor, c2: torch.Tensor, scale: float = 1.0) -> DiffExpr3:
    x = _zonotope(c1, scale)
    y = _zonotope(c2, scale)
    return DiffExpr3(x, y, x - y)


def _sample_pairs(c1: torch.Tensor, c2: torch.Tensor, n: int, scale: float):
    s1 = c1 + (torch.rand(n, *c1.shape) * 2 - 1) * scale
    s2 = c2 + (torch.rand(n, *c2.shape) * 2 - 1) * scale
    return s1, s2


def _check_pair_sound(out: DiffExpr2, model, s1, s2, tol: float = 1e-4):
    ub_x, lb_x = out.x.ublb()
    ub_y, lb_y = out.y.ublb()
    with torch.no_grad():
        f1 = model(s1)
        f2 = model(s2)
    assert (f1 <= ub_x + tol).all(), (f1 - ub_x).max().item()
    assert (f1 >= lb_x - tol).all(), (lb_x - f1).max().item()
    assert (f2 <= ub_y + tol).all(), (f2 - ub_y).max().item()
    assert (f2 >= lb_y - tol).all(), (lb_y - f2).max().item()
    # Induced diff interval
    d = f1 - f2
    ub_d = ub_x - lb_y
    lb_d = lb_x - ub_y
    assert (d <= ub_d + tol).all(), (d - ub_d).max().item()
    assert (d >= lb_d - tol).all(), (lb_d - d).max().item()


# =====================================================================
# Construction: ZonoHexGate shape/flags bookkeeping
# =====================================================================

def test_gate_shape_and_children():
    x = _zonotope(torch.zeros(3))
    y = _zonotope(torch.zeros(3))
    d = x - y
    gate = ZonoHexGate(x, y, d)
    assert gate.shape == (x.shape, y.shape)
    assert gate.children == (x, y, d)
    # dummy 3rd output flag so len(flags) == len(children)
    assert len(gate.flags) == 3


def test_gate_shape_mismatch_raises():
    x = _zonotope(torch.zeros(3))
    y = _zonotope(torch.zeros(4))
    d = _zonotope(torch.zeros(3))
    with pytest.raises(AssertionError):
        ZonoHexGate(x, y, d)


def test_expr3_to_expr2_passthrough_for_plain_expr():
    """Non-DiffExpr3 inputs should pass through the converter unchanged."""
    e = _zonotope(torch.zeros(3))
    assert expr3_to_expr2(e) is e


def test_expr3_to_expr2_converts_to_pair():
    c1, c2 = torch.randn(3), torch.randn(3)
    triple = _make_triple(c1, c2)
    out = expr3_to_expr2(triple)
    assert isinstance(out, DiffExpr2)
    assert out.x.shape == triple.x.shape
    assert out.y.shape == triple.y.shape


# =====================================================================
# Soundness
# =====================================================================

def test_sound_single_linear():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    op = hex_interpret(onnx_export(model, [4]))

    c1, c2 = torch.randn(4), torch.randn(4)
    scale = 0.5
    out = op(_make_triple(c1, c2, scale))

    s1, s2 = _sample_pairs(c1, c2, n=2000, scale=scale)
    _check_pair_sound(out, model, s1, s2)


def test_sound_relu_mlp():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    op = hex_interpret(onnx_export(model, [4]))

    c1, c2 = torch.randn(4), torch.randn(4)
    scale = 0.3
    out = op(_make_triple(c1, c2, scale))

    s1, s2 = _sample_pairs(c1, c2, n=2000, scale=scale)
    _check_pair_sound(out, model, s1, s2)


def test_sound_deeper_relu_mlp():
    torch.manual_seed(2)
    model = nn.Sequential(
        nn.Linear(5, 8), nn.ReLU(),
        nn.Linear(8, 6), nn.ReLU(),
        nn.Linear(6, 4),
    )
    op = hex_interpret(onnx_export(model, [5]))

    c1, c2 = torch.randn(5), torch.randn(5)
    scale = 0.2
    out = op(_make_triple(c1, c2, scale))

    s1, s2 = _sample_pairs(c1, c2, n=2000, scale=scale)
    _check_pair_sound(out, model, s1, s2)


def test_equal_inputs_tight_diff():
    """When x == y (same center, shared eps), the induced diff interval should
    contain zero (the true diff is identically zero)."""
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    op = hex_interpret(onnx_export(model, [4]))

    c = torch.randn(4)
    x = _zonotope(c, 0.3)
    # Same expression for both sides → exact diff = 0
    triple = DiffExpr3(x, x, x - x)
    out = op(triple)

    ub_x, lb_x = out.x.ublb()
    ub_y, lb_y = out.y.ublb()
    # Diff interval [lb_x - ub_y, ub_x - lb_y] must contain 0.
    assert (lb_x - ub_y <= 1e-5).all()
    assert (ub_x - lb_y >= -1e-5).all()


# =====================================================================
# Fallback: plain Expr inputs should still work
# =====================================================================

def test_plain_expr_passthrough():
    """Feeding a plain Expr (not a DiffExpr3) should interpret as standard
    zonotope propagation and yield a plain Expr-compatible result."""
    torch.manual_seed(4)
    model = nn.Linear(4, 3)
    op = hex_interpret(onnx_export(model, [4]))

    c = torch.randn(4)
    z = _zonotope(c, 0.2)
    out = op(z)

    # out should be an Expr (not a DiffExpr2); its bounds should contain f(s).
    ub, lb = out.ublb()
    s = c + (torch.rand(1500, 4) * 2 - 1) * 0.2
    with torch.no_grad():
        f = model(s)
    assert (f <= ub + 1e-4).all()
    assert (f >= lb - 1e-4).all()
