"""Unit tests for the heaviside_pruning differential lineariser."""

import torch

from boundlab import expr
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.diff.zono3.default.heaviside import _linearize_hsx, diff_heaviside_pruning_handler


def test_linearize_hsx_always_zero():
    ls = torch.tensor([-2.0])
    us = torch.tensor([-1.0])
    lx = torch.tensor([-1.0])
    ux = torch.tensor([-0.5])

    w_s, w_x, bias, err = _linearize_hsx(ls, us, lx, ux)

    assert torch.allclose(w_s, torch.zeros_like(ls))
    assert torch.allclose(w_x, torch.zeros_like(ls))
    assert torch.allclose(bias, torch.zeros_like(ls))
    assert torch.allclose(err, torch.zeros_like(ls))


def test_linearize_hsx_case2_ux_nonpos():
    ls = torch.tensor([-1.0])
    us = torch.tensor([1.0])
    lx = torch.tensor([-2.0])
    ux = torch.tensor([-1.0])

    w_s, w_x, bias, err = _linearize_hsx(ls, us, lx, ux)

    # lam = max(lx / -ls, ux / us) = max(-2, -1) = -1
    assert torch.allclose(w_s, torch.tensor([-1.0]), atol=1e-6)
    assert torch.allclose(w_x, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(bias, torch.tensor([-1.0]), atol=1e-6)
    assert torch.allclose(err, torch.tensor([1.0]), atol=1e-6)


def test_linearize_hsx_case3_ux_pos():
    ls = torch.tensor([-1.0])
    us = torch.tensor([1.0])
    lx = torch.tensor([-1.0])
    ux = torch.tensor([2.0])

    w_s, w_x, bias, err = _linearize_hsx(ls, us, lx, ux)

    # lam = min(ux/(ux-lx), -lx/(ux-lx)) = min(2/3, 1/3) = 1/3
    assert torch.allclose(w_s, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(w_x, torch.tensor([1.0 / 3.0]), atol=1e-6)
    assert torch.allclose(bias, torch.tensor([1.0 / 3.0]), atol=1e-6)
    assert torch.allclose(err, torch.tensor([1.0]), atol=1e-6)


def test_handler_promotes_tensor_scores_and_data():
    scores = torch.ones(4)
    base = expr.ConstVal(torch.zeros(4)) + 0.1 * expr.LpEpsilon([4])
    data = DiffExpr3(base, base, base * 0)

    out = diff_heaviside_pruning_handler(scores, data)

    assert isinstance(out, DiffExpr3)
    assert out.x.shape == scores.shape
    assert out.y.shape == scores.shape
    assert out.diff.shape == scores.shape


def test_handler_accepts_diffexpr2_inputs():
    s_x = expr.ConstVal(torch.tensor([0.5]))
    s_y = expr.ConstVal(torch.tensor([-0.5]))
    scores = DiffExpr2(s_x, s_y)

    d_x = expr.ConstVal(torch.tensor([1.0]))
    d_y = expr.ConstVal(torch.tensor([0.2]))
    data = DiffExpr2(d_x, d_y)

    out = diff_heaviside_pruning_handler(scores, data)

    assert isinstance(out, DiffExpr3)
    assert out.x.shape == torch.Size([1])
    assert out.y.shape == torch.Size([1])
    assert out.diff.shape == torch.Size([1])


def test_linearize_hsx_sampling_soundness():
    torch.manual_seed(0)
    # Random box bounds
    ls = torch.rand(32) * 2 - 2   # [-2,0)
    us = torch.rand(32) * 2       # [0,2)
    lx = torch.rand(32) * 4 - 2   # [-2,2)
    ux = lx + torch.rand(32) * 2  # ensure ux >= lx

    w_s, w_x, bias, err = _linearize_hsx(ls, us, lx, ux)

    # Monte Carlo sampling to check soundness
    num_samples = 64
    s_samples = ls[:, None] + torch.rand(32, num_samples) * (us - ls)[:, None]
    x_samples = lx[:, None] + torch.rand(32, num_samples) * (ux - lx)[:, None]
    h = (s_samples >= 0).float()
    true_vals = h * x_samples
    approx = w_s[:, None] * s_samples + w_x[:, None] * x_samples + bias[:, None]
    gap = (true_vals - approx).abs()
    bound = err[:, None] + 1e-6

    assert torch.all(gap <= bound)
