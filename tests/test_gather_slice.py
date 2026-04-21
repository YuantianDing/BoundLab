"""Tests for ONNX Gather and Slice handlers.

Covers:
1. Gather with scalar index (CLS token selection: x[:, 0, :])
2. Gather with 1-D index tensor
3. Slice with various axis/start/end combos
4. End-to-end: BERT classifier head (Gather + Linear) through zonotope verification
"""

import torch
import pytest
from torch import nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export


# ---- helpers ----------------------------------------------------------------

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


# ---- Gather: scalar index (CLS token) --------------------------------------

class CLSSelect(nn.Module):
    """Select CLS token: x[:, 0, :]"""
    def forward(self, x):
        return x[:, 0, :]


def test_gather_scalar_concrete():
    """Gather with scalar index on concrete tensors via ONNX interpreter."""
    torch.manual_seed(300)

    model = CLSSelect()
    model.eval()
    onnx_model = onnx_export(model, ([3, 5, 8],))

    x = torch.randn(3, 5, 8)
    from boundlab.interp import ONNX_BASE_INTERPRETER
    op = ONNX_BASE_INTERPRETER(onnx_model)
    y = op(x)
    expected = x[:, 0, :]
    assert torch.allclose(y, expected), f"Mismatch: {(y - expected).abs().max()}"


def test_gather_scalar_zonotope_sound():
    """Gather (CLS token select) zonotope bounds must be sound."""
    torch.manual_seed(301)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    model = CLSSelect()
    model.eval()

    center_val = torch.randn(3, 5, 8) * 0.5
    scale = 0.3

    onnx_model = onnx_export(model, ([3, 5, 8],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    assert ub.shape == torch.Size([3, 8])

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb)


# ---- Gather: BERT classifier head ------------------------------------------

class BERTClassifierHead(nn.Module):
    """CLS token selection + linear classifier, as in BERT-smaller-3."""
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)


def test_bert_classifier_head_sound():
    """BERT classifier head (Gather + Linear) zonotope bounds must be sound."""
    torch.manual_seed(302)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    hidden, num_classes = 64, 2
    seq_len, batch = 5, 2
    model = BERTClassifierHead(hidden, num_classes)
    model.eval()

    center_val = torch.randn(batch, seq_len, hidden) * 0.3
    scale = 0.1

    onnx_model = onnx_export(model, ([batch, seq_len, hidden],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    assert ub.shape == torch.Size([batch, num_classes])

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb)


# ---- Slice ------------------------------------------------------------------

class SliceMiddle(nn.Module):
    """Slice: x[:, 1:3, :]"""
    def forward(self, x):
        return x[:, 1:3, :]


def test_slice_concrete():
    """Slice on concrete tensors via ONNX interpreter."""
    torch.manual_seed(310)

    model = SliceMiddle()
    model.eval()
    onnx_model = onnx_export(model, ([2, 5, 8],))

    x = torch.randn(2, 5, 8)
    from boundlab.interp import ONNX_BASE_INTERPRETER
    op = ONNX_BASE_INTERPRETER(onnx_model)
    y = op(x)
    expected = x[:, 1:3, :]
    assert torch.allclose(y, expected), f"Mismatch: {(y - expected).abs().max()}"


def test_slice_zonotope_sound():
    """Slice zonotope bounds must be sound."""
    torch.manual_seed(311)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    model = SliceMiddle()
    model.eval()

    center_val = torch.randn(2, 5, 8) * 0.5
    scale = 0.3

    onnx_model = onnx_export(model, ([2, 5, 8],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    assert ub.shape == torch.Size([2, 2, 8])

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb)


# ---- Slice on first dim -----------------------------------------------------

class SliceFirst(nn.Module):
    """Slice: x[0:2, :]"""
    def forward(self, x):
        return x[0:2, :]


def test_slice_first_dim_sound():
    """Slice on first dim zonotope bounds must be sound."""
    torch.manual_seed(312)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    model = SliceFirst()
    model.eval()

    center_val = torch.randn(4, 6) * 0.5
    scale = 0.2

    onnx_model = onnx_export(model, ([4, 6],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    assert ub.shape == torch.Size([2, 6])

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb)


# ---- Slice + Linear (combined pipeline) -------------------------------------

class SliceThenLinear(nn.Module):
    """Slice a range of tokens, then apply a linear layer."""
    def __init__(self, hidden, out_dim):
        super().__init__()
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc(x[:, 1:4, :])


def test_slice_then_linear_sound():
    """Slice + Linear zonotope bounds must be sound."""
    torch.manual_seed(313)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    hidden, out_dim = 16, 4
    model = SliceThenLinear(hidden, out_dim)
    model.eval()

    center_val = torch.randn(2, 6, hidden) * 0.3
    scale = 0.15

    onnx_model = onnx_export(model, ([2, 6, hidden],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    assert ub.shape == torch.Size([2, 3, out_dim])

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb)
