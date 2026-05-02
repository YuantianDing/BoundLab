"""Tests for `boundlab.gradlin`.

The simplified pipeline works in three steps:
1. Sample a batched trapezoid.
2. Fit ``lam_x, lam_y`` from the samples.
3. Search the region for lower/upper residual bounds.
"""

from __future__ import annotations

import sys
import types

import torch

from boundlab.gradlin import gradlin
from boundlab.gradlin._core import _fit_lam_gurobi


class _FakeGurobiExpr:
    def __init__(self, name=None, value=0.0):
        self.name = name
        self.X = value

    def __mul__(self, other):
        return _FakeGurobiExpr(self.name, self.X)

    __rmul__ = __mul__

    def __neg__(self):
        return _FakeGurobiExpr(self.name, self.X)

    def __sub__(self, other):
        return _FakeGurobiExpr(self.name, self.X)

    def __rsub__(self, other):
        return _FakeGurobiExpr(self.name, self.X)

    def __le__(self, other):
        return ("le", self.name, other)


class _FakeGurobiVar(_FakeGurobiExpr):
    def __init__(self, name, value):
        super().__init__(name, value)


class _FakeGurobiModel:
    def __init__(self, status=2, value_map=None, calls=None):
        self.Params = types.SimpleNamespace(OutputFlag=None, LogToConsole=None)
        self.Status = status
        self._value_map = value_map or {}
        self._calls = calls if calls is not None else {}

    def addVar(self, lb, name):
        return _FakeGurobiVar(name, self._value_map.get(name, 0.0))

    def addConstr(self, expr):
        self._calls.setdefault("constraints", []).append(expr)

    def setObjective(self, obj, sense):
        self._calls["objective"] = (obj, sense)

    def optimize(self):
        self._calls["optimized"] = True


def _sample_trapezoid(
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
    n: int,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x_lo = torch.maximum(lx, ly + ld)
    x_hi = torch.minimum(ux, uy + ud)
    x_w = (x_hi - x_lo).clamp_min(0.0)
    rx = torch.rand(n, *lx.shape, generator=g)
    x = x_lo.unsqueeze(0) + rx * x_w.unsqueeze(0)
    y_lo = torch.maximum(ly.unsqueeze(0), x - ud.unsqueeze(0))
    y_hi = torch.minimum(uy.unsqueeze(0), x - ld.unsqueeze(0))
    ry = torch.rand(x.shape, generator=g)
    y = y_lo + ry * (y_hi - y_lo).clamp_min(0.0)
    return x, y


def _assert_sound(f, lam, L, U, x, y, *, tol=3e-2):
    resid = f(x) - f(y) - lam[..., 0].unsqueeze(0) * x - lam[..., 1].unsqueeze(0) * y
    assert (resid >= L.unsqueeze(0) - tol).all(), (
        f"Lower bound violated: max deficit = {(L.unsqueeze(0) - resid).max():.6f}"
    )
    assert (resid <= U.unsqueeze(0) + tol).all(), (
        f"Upper bound violated: max excess = {(resid - U.unsqueeze(0)).max():.6f}"
    )


def test_gradlin_exponential():
    lx = torch.tensor([-1.0, -0.5, 0.0, 0.2])
    ux = torch.tensor([0.0, 0.5, 1.0, 0.9])
    ly = torch.tensor([-0.5, -0.3, 0.1, 0.0])
    uy = torch.tensor([0.5, 0.4, 0.7, 0.6])
    ld = torch.tensor([-0.8, -0.5, -0.5, -0.2])
    ud = torch.tensor([0.8, 0.5, 0.6, 0.6])

    lam, L, U = gradlin(
        torch.exp, lx, ux, ly, uy, ld, ud, num_samples=1024, num_starts=48, iters=60, lr=0.05
    )

    assert lam.shape == (4, 2)
    assert L.shape == (4,) and U.shape == (4,)

    x, y = _sample_trapezoid(lx, ux, ly, uy, ld, ud, n=2048)
    _assert_sound(torch.exp, lam, L, U, x, y)

    resid = torch.exp(x) - torch.exp(y)
    zero_width = resid.amax(dim=0) - resid.amin(dim=0)
    fit_width = U - L
    assert (fit_width <= zero_width + 5e-2).all()
    assert (fit_width < zero_width - 1e-2).any()


def test_gradlin_tanh():
    lx = torch.tensor([-1.0, -0.5, -0.8, -1.5])
    ux = torch.tensor([1.0, 0.5, 1.2, 0.3])
    ly = torch.tensor([-0.5, -1.0, -0.4, -1.0])
    uy = torch.tensor([0.5, 1.0, 1.0, 1.2])
    ld = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    ud = torch.tensor([10.0, 10.0, 10.0, 10.0])

    lam, L, U = gradlin(
        torch.tanh, lx, ux, ly, uy, ld, ud, num_samples=1024, num_starts=48, iters=60, lr=0.05
    )

    x, y = _sample_trapezoid(lx, ux, ly, uy, ld, ud, n=2048)
    _assert_sound(torch.tanh, lam, L, U, x, y)


def test_gradlin_square():
    lx = torch.tensor([-1.0, 0.0, -2.0, 0.5])
    ux = torch.tensor([1.0, 2.0, 0.0, 1.5])
    ly = torch.tensor([-1.0, -1.0, 0.0, -1.0])
    uy = torch.tensor([1.0, 1.0, 2.0, 0.5])
    ld = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    ud = torch.tensor([10.0, 10.0, 10.0, 10.0])

    def f(x):
        return x * x

    lam, L, U = gradlin(f, lx, ux, ly, uy, ld, ud, num_samples=1024, num_starts=48, iters=60, lr=0.05)

    x, y = _sample_trapezoid(lx, ux, ly, uy, ld, ud, n=2048)
    _assert_sound(f, lam, L, U, x, y)


def test_gradlin_square_symmetric():
    lx = torch.tensor([-1.0])
    ux = torch.tensor([1.0])
    ly = torch.tensor([-1.0])
    uy = torch.tensor([1.0])
    ld = torch.tensor([-10.0])
    ud = torch.tensor([10.0])

    def f(x):
        return x * x

    lam, L, U = gradlin(f, lx, ux, ly, uy, ld, ud, num_samples=1024, num_starts=48, iters=80, lr=0.05)

    assert lam.shape == (1, 2)
    assert torch.isfinite(lam).all()
    assert torch.isfinite(L).all() and torch.isfinite(U).all()


def test_gurobi_backend_is_used(monkeypatch):
    calls = {}
    class FakeModel(_FakeGurobiModel):
        def __init__(self):
            super().__init__(status=2, value_map={"lamx": 1.5, "lamy": -2.0, "t": 0.25}, calls=calls)

    fake_gp = types.ModuleType("gurobipy")
    fake_gp.Model = FakeModel
    fake_gp.GRB = types.SimpleNamespace(INFINITY=float("inf"), MINIMIZE="min", OPTIMAL=2)

    monkeypatch.setitem(sys.modules, "gurobipy", fake_gp)

    fx = torch.tensor([[2.0], [3.0]])
    fy = torch.tensor([[0.5], [1.0]])
    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[0.25], [0.5]])

    lam = _fit_lam_gurobi(fx, fy, x, y)

    assert torch.allclose(lam, torch.tensor([[1.5, -2.0]]))
    assert calls["optimized"] is True
    assert calls["objective"][1] == "min"


def test_gradlin_uses_gurobi_end_to_end(monkeypatch):
    class FakeModel(_FakeGurobiModel):
        def __init__(self):
            super().__init__(status=2, value_map={"lamx": 0.0, "lamy": 0.0, "t": 0.0})

    fake_gp = types.ModuleType("gurobipy")
    fake_gp.Model = FakeModel
    fake_gp.GRB = types.SimpleNamespace(INFINITY=float("inf"), MINIMIZE="min", OPTIMAL=2)
    monkeypatch.setitem(sys.modules, "gurobipy", fake_gp)

    lx = torch.tensor([-1.0])
    ux = torch.tensor([1.0])
    ly = torch.tensor([-1.0])
    uy = torch.tensor([1.0])
    ld = torch.tensor([-2.0])
    ud = torch.tensor([2.0])

    lam, L, U = gradlin(
        lambda x: torch.zeros_like(x),
        lx,
        ux,
        ly,
        uy,
        ld,
        ud,
        num_samples=8,
        num_starts=4,
        iters=0,
        use_gurobi=True,
    )

    assert torch.allclose(lam, torch.zeros_like(lam))
    assert torch.allclose(L, torch.zeros_like(L))
    assert torch.allclose(U, torch.zeros_like(U))
