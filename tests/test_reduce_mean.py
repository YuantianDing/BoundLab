"""Tests for ReduceMean operator and LayerNormNoVar support.

Covers:
1. ReduceMeanOp LinearOp correctness (forward, backward, vforward, vbackward)
2. Expr.mean() zonotope soundness
3. End-to-end LayerNormNoVar through ONNX export + zonotope verification
"""

import torch
import pytest
from torch import nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.linearop import ReduceMeanOp
from boundlab.interp.onnx import onnx_export


# ---- ReduceMeanOp unit tests -----------------------------------------------

class TestReduceMeanOp:

    def test_forward_single_dim(self):
        op = ReduceMeanOp(torch.Size([3, 4]), dims=(1,), keepdim=False)
        x = torch.randn(3, 4)
        assert torch.allclose(op.forward(x), x.mean(dim=1))
        assert op.output_shape == torch.Size([3])

    def test_forward_keepdim(self):
        op = ReduceMeanOp(torch.Size([3, 4]), dims=(1,), keepdim=True)
        x = torch.randn(3, 4)
        assert torch.allclose(op.forward(x), x.mean(dim=1, keepdim=True))
        assert op.output_shape == torch.Size([3, 1])

    def test_forward_last_dim(self):
        op = ReduceMeanOp(torch.Size([2, 3, 5]), dims=(-1,), keepdim=True)
        x = torch.randn(2, 3, 5)
        assert torch.allclose(op.forward(x), x.mean(dim=-1, keepdim=True))

    def test_forward_multiple_dims(self):
        op = ReduceMeanOp(torch.Size([2, 3, 4]), dims=(0, 2), keepdim=False)
        x = torch.randn(2, 3, 4)
        assert torch.allclose(op.forward(x), x.mean(dim=(0, 2)))

    def test_backward_adjoint(self):
        """backward must be the adjoint of forward: <forward(x), y> == <x, backward(y)>"""
        for keepdim in [False, True]:
            op = ReduceMeanOp(torch.Size([3, 4]), dims=(1,), keepdim=keepdim)
            x = torch.randn(3, 4)
            y = torch.randn(op.output_shape)
            lhs = (op.forward(x) * y).sum()
            rhs = (x * op.backward(y)).sum()
            assert torch.allclose(lhs, rhs, atol=1e-6), f"keepdim={keepdim}: {lhs} != {rhs}"

    def test_backward_adjoint_3d(self):
        op = ReduceMeanOp(torch.Size([2, 3, 5]), dims=(-1,), keepdim=True)
        x = torch.randn(2, 3, 5)
        y = torch.randn(op.output_shape)
        lhs = (op.forward(x) * y).sum()
        rhs = (x * op.backward(y)).sum()
        assert torch.allclose(lhs, rhs, atol=1e-6)

    def test_vforward(self):
        op = ReduceMeanOp(torch.Size([3, 4]), dims=(1,), keepdim=True)
        # x has extra trailing dims (batch of generators)
        x = torch.randn(3, 4, 7)
        out = op.vforward(x)
        assert out.shape == torch.Size([3, 1, 7])
        # Each slice should match forward
        for i in range(7):
            assert torch.allclose(out[..., i], op.forward(x[..., i]))

    def test_vbackward(self):
        op = ReduceMeanOp(torch.Size([3, 4]), dims=(1,), keepdim=True)
        grad = torch.randn(5, 3, 1)  # leading batch dim
        out = op.vbackward(grad)
        assert out.shape == torch.Size([5, 3, 4])
        for i in range(5):
            assert torch.allclose(out[i], op.backward(grad[i]))


# ---- Expr.mean() zonotope soundness ----------------------------------------

def _make_input(center_val, scale=1.0):
    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon(list(center_val.shape))
    return center + eps * scale if scale != 1.0 else center + eps


def _sample_inputs(center, scale, n=2000):
    noise = torch.rand(n, *center.shape) * 2 - 1
    return center.unsqueeze(0) + scale * noise


def _check_bounds(outputs, ub, lb, tol=1e-4):
    assert (ub >= lb - tol).all(), f"UB < LB: max violation = {(lb - ub).max():.6f}"
    assert (outputs <= ub.unsqueeze(0) + tol).all(), f"UB violated: {(outputs - ub.unsqueeze(0)).max():.6f}"
    assert (outputs >= lb.unsqueeze(0) - tol).all(), f"LB violated: {(lb.unsqueeze(0) - outputs).max():.6f}"


def test_expr_mean_sound():
    """Zonotope bounds for Expr.mean() must contain all sampled outputs."""
    torch.manual_seed(200)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    center_val = torch.randn(4, 8)
    scale = 0.5
    x_expr = _make_input(center_val, scale=scale)
    y_expr = x_expr.mean(dim=-1, keepdim=True)

    ub, lb = y_expr.ublb()
    assert ub.shape == torch.Size([4, 1])

    samples = _sample_inputs(center_val, scale, n=3000)
    outputs = samples.mean(dim=-1, keepdim=True)
    _check_bounds(outputs, ub, lb)


def test_expr_mean_no_keepdim_sound():
    torch.manual_seed(201)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    center_val = torch.randn(3, 5)
    scale = 0.3
    x_expr = _make_input(center_val, scale=scale)
    y_expr = x_expr.mean(dim=1)

    ub, lb = y_expr.ublb()
    assert ub.shape == torch.Size([3])

    samples = _sample_inputs(center_val, scale, n=3000)
    outputs = samples.mean(dim=-1)
    _check_bounds(outputs, ub, lb)


# ---- LayerNormNoVar end-to-end via ONNX ------------------------------------

class LayerNormNoVar(nn.Module):
    """LayerNorm without variance normalization (as used in DeepT BERT)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias


class LayerNormNoVarFFN(nn.Module):
    """A small FFN block with LayerNormNoVar + residual, matching BERT structure."""
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate, hidden)
        self.norm = LayerNormNoVar(hidden)

    def forward(self, x):
        h = self.fc2(self.relu(self.fc1(x)))
        return self.norm(h + x)


def test_layernorm_novar_sound():
    """LayerNormNoVar exported to ONNX and verified via zonotopes."""
    torch.manual_seed(202)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    hidden = 16
    model = LayerNormNoVar(hidden)
    model.eval()

    center_val = torch.randn(3, hidden) * 0.5
    scale = 0.2

    onnx_model = onnx_export(model, ([3, hidden],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb)


def test_layernorm_novar_ffn_sound():
    """Full FFN + LayerNormNoVar + residual, matching BERT FFN block structure."""
    torch.manual_seed(203)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    hidden, intermediate = 16, 32
    model = LayerNormNoVarFFN(hidden, intermediate)
    model.eval()

    center_val = torch.randn(3, hidden) * 0.3
    scale = 0.1

    onnx_model = onnx_export(model, ([3, hidden],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb)
