"""Tests for N-D bilinear matmul, BERT model export, and robustness verification.

Covers:
1. 4-D bilinear matmul (multi-head attention Q @ K^T)
2. Full BERT-smaller-3 model ONNX export + zonotope verification
3. Robustness verification property (argmax stability)
"""

import math
import torch
import pytest
from torch import nn

import boundlab.expr as expr
from boundlab.expr._core import Expr
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


# ---- 4-D bilinear matmul (multi-head attention) ----------------------------

def test_bilinear_matmul_4d_sound():
    """4-D bilinear matmul (B, H, S, S) for multi-head attention scores."""
    torch.manual_seed(400)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    B, H, S, D = 2, 4, 3, 16
    center_Q = torch.randn(B, H, S, D) * 0.3
    center_K = torch.randn(B, H, S, D) * 0.3
    scale = 0.05

    Q_expr = _make_input(center_Q, scale=scale)
    K_expr = _make_input(center_K, scale=scale)

    scores = zono.bilinear_matmul(Q_expr, K_expr.transpose(-2, -1))
    ub, lb = scores.ublb()
    assert ub.shape == torch.Size([B, H, S, S])

    for _ in range(3000):
        nQ = (torch.rand_like(center_Q) * 2 - 1) * scale
        nK = (torch.rand_like(center_K) * 2 - 1) * scale
        out = (center_Q + nQ) @ (center_K + nK).transpose(-2, -1)
        assert (out <= ub + 1e-4).all(), f"UB violated: {(out - ub).max():.6f}"
        assert (out >= lb - 1e-4).all(), f"LB violated: {(lb - out).max():.6f}"


def test_bilinear_matmul_4d_context_sound():
    """4-D bilinear matmul for attn @ V context computation."""
    torch.manual_seed(401)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    B, H, S, D = 2, 4, 3, 8
    center_attn = torch.randn(B, H, S, S).abs()  # positive-ish attention weights
    center_attn = center_attn / center_attn.sum(-1, keepdim=True)  # normalize
    center_V = torch.randn(B, H, S, D) * 0.3
    scale = 0.02

    attn_expr = _make_input(center_attn, scale=scale)
    V_expr = _make_input(center_V, scale=scale)

    context = zono.bilinear_matmul(attn_expr, V_expr)
    ub, lb = context.ublb()
    assert ub.shape == torch.Size([B, H, S, D])

    for _ in range(3000):
        na = (torch.rand_like(center_attn) * 2 - 1) * scale
        nv = (torch.rand_like(center_V) * 2 - 1) * scale
        out = (center_attn + na) @ (center_V + nv)
        assert (out <= ub + 1e-4).all(), f"UB violated: {(out - ub).max():.6f}"
        assert (out >= lb - 1e-4).all(), f"LB violated: {(lb - out).max():.6f}"


# ---- BERT-smaller-3 model ---------------------------------------------------

class LayerNormNoVar(nn.Module):
    """LayerNorm without variance normalization (DeepT)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias


class BERTSelfAttention(nn.Module):
    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

    def forward(self, x):
        S = x.shape[0]
        Q = self.query(x).view(S, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = self.key(x).view(S, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = self.value(x).view(S, self.num_heads, self.head_dim).permute(1, 0, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (H, S, D)
        context = context.permute(1, 0, 2).reshape(S, -1)  # (S, hidden)
        return context


class BERTSelfOutput(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.norm = LayerNormNoVar(hidden)

    def forward(self, context, residual):
        return self.norm(self.dense(context) + residual)


class BERTFFN(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.dense1 = nn.Linear(hidden, intermediate)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(intermediate, hidden)
        self.norm = LayerNormNoVar(hidden)

    def forward(self, x):
        h = self.dense2(self.relu(self.dense1(x)))
        return self.norm(h + x)


class BERTBlock(nn.Module):
    def __init__(self, hidden, intermediate, num_heads, head_dim):
        super().__init__()
        self.attention = BERTSelfAttention(hidden, num_heads, head_dim)
        self.self_output = BERTSelfOutput(hidden)
        self.ffn = BERTFFN(hidden, intermediate)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.self_output(attn_out, x)
        x = self.ffn(x)
        return x


class BERTSmaller3(nn.Module):
    """BERT-smaller-3: 3-layer BERT for SST-2 (DeepT config).

    Config: hidden=64, intermediate=128, heads=4, head_dim=16, layers=3,
            relu activation, LayerNormNoVar, 2 output classes.
    """
    def __init__(self, hidden=64, intermediate=128, num_heads=4, head_dim=16,
                 num_layers=3, num_classes=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            BERTBlock(hidden, intermediate, num_heads, head_dim)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (S, hidden) — post-embedding tensor (no batch dim)
        for block in self.blocks:
            x = block(x)
        # CLS token classification
        cls_token = x[0]  # (hidden,)
        return self.classifier(cls_token)  # (num_classes,)


# ---- BERT ONNX export + zonotope tests -------------------------------------

def test_bert_single_block_sound():
    """Single BERT block through ONNX + zonotope is sound."""
    torch.manual_seed(500)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    hidden, intermediate, num_heads, head_dim = 64, 128, 4, 16
    seq_len = 4

    model = BERTBlock(hidden, intermediate, num_heads, head_dim)
    model.eval()

    center_val = torch.randn(seq_len, hidden) * 0.1
    scale = 0.01

    onnx_model = onnx_export(model, ([seq_len, hidden],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    assert ub.shape == torch.Size([seq_len, hidden])

    samples = _sample_inputs(center_val, scale, n=1000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb, tol=1e-3)


def test_bert_full_model_sound():
    """Full BERT-smaller-3 through ONNX + zonotope is sound."""
    torch.manual_seed(501)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    model = BERTSmaller3()
    model.eval()

    seq_len = 8
    hidden = 64
    center_val = torch.randn(seq_len, hidden) * 0.05
    scale = 0.005  # small perturbation for 3-layer model

    onnx_model = onnx_export(model, ([seq_len, hidden],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    assert ub.shape == torch.Size([2])  # 2 classes

    samples = _sample_inputs(center_val, scale, n=1000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb, tol=1e-3)


# ---- Robustness verification property --------------------------------------

def test_robustness_verification():
    """Demonstrate the full robustness verification pipeline.

    Given a concrete input and its true label, verify that the model's
    prediction cannot flip under L∞ perturbation of radius ε.
    """
    torch.manual_seed(502)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    model = BERTSmaller3()
    model.eval()

    seq_len = 4
    hidden = 64
    center_val = torch.randn(seq_len, hidden) * 0.05

    # Get the model's prediction on the clean input
    with torch.no_grad():
        clean_logits = model(center_val)
        true_label = int(clean_logits.argmax().item())

    # Try to certify robustness at a small ε
    scale = 0.002
    onnx_model = onnx_export(model, ([seq_len, hidden],))
    op = zono.interpret(onnx_model)
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)

    # Verification property: logit_diff = logit[true] - logit[other] > 0
    # Use narrow to get (1,) slices instead of scalar indexing
    other_label = 1 - true_label
    logit_diff = y_expr.narrow(0, true_label, 1) - y_expr.narrow(0, other_label, 1)
    lb_diff = prop.lb(logit_diff).item()

    if lb_diff > 0:
        # Verified: no perturbation can flip the prediction
        # Confirm by sampling
        samples = _sample_inputs(center_val, scale, n=1000)
        with torch.no_grad():
            preds = torch.stack([model(s) for s in samples]).argmax(dim=-1)
        assert (preds == true_label).all(), "Sampling contradicts verification!"
    else:
        # Not verified at this ε — that's also a valid outcome
        pass

    # At least check that the pipeline ran without errors
    assert isinstance(lb_diff, float)
