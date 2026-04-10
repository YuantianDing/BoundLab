"""BERT certification tests.

Tests:
1. 4D bilinear matmul (batched attention) soundness
2. Single BERT block (LayerNormNoVar + Attention + FFN + residual) soundness
3. Full BERT model (embedding + N blocks + classifier) soundness
4. Robustness verification on BERT
5. Differential verification on BERT
"""

import copy
import math
import warnings

import torch
from torch import nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

warnings.filterwarnings("ignore")


# ---- BERT components (DeepT small model) ------------------------------------

class LayerNormNoVar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias


class BertAttention(nn.Module):
    def __init__(self, hidden=64, num_heads=4, head_dim=16):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)
        self.query = nn.Linear(hidden, inner)
        self.key = nn.Linear(hidden, inner)
        self.value = nn.Linear(hidden, inner)
        self.out_proj = nn.Linear(inner, hidden)
    def forward(self, x):
        S, _ = x.shape
        h = self.num_heads
        Q = self.query(x).reshape(S, h, -1).permute(1, 0, 2)
        K = self.key(x).reshape(S, h, -1).permute(1, 0, 2)
        V = self.value(x).reshape(S, h, -1).permute(1, 0, 2)
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        attn = scores.softmax(dim=-1)
        context = (attn @ V).permute(1, 0, 2).reshape(S, -1)
        return self.out_proj(context)


class BertBlock(nn.Module):
    def __init__(self, hidden=64, intermediate=64, num_heads=4, head_dim=16):
        super().__init__()
        self.attn_norm = LayerNormNoVar(hidden)
        self.attn = BertAttention(hidden, num_heads, head_dim)
        self.ff_norm = LayerNormNoVar(hidden)
        self.ff1 = nn.Linear(hidden, intermediate)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(intermediate, hidden)
    def forward(self, x):
        x = self.attn(self.attn_norm(x)) + x
        x = self.ff2(self.relu(self.ff1(self.ff_norm(x)))) + x
        return x


class BertModel(nn.Module):
    def __init__(self, hidden=64, intermediate=64, num_heads=4, head_dim=16,
                 num_layers=1, num_classes=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            BertBlock(hidden, intermediate, num_heads, head_dim)
            for _ in range(num_layers)
        ])
        self.classifier_norm = LayerNormNoVar(hidden)
        self.classifier = nn.Linear(hidden, num_classes)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        cls = x[0]  # CLS token
        return self.classifier(self.classifier_norm(cls))


# ---- Helpers ----------------------------------------------------------------

def _make_input(center, eps):
    return expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps

def _sample_inputs(center, eps, n=2000):
    return center.unsqueeze(0) + eps * (torch.rand(n, *center.shape) * 2 - 1)

def _check_bounds(outputs, ub, lb, tol=1e-4):
    assert (ub >= lb - tol).all(), f"UB < LB: {(lb - ub).max():.6f}"
    assert (outputs <= ub.unsqueeze(0) + tol).all(), f"UB violated: {(outputs - ub.unsqueeze(0)).max():.6f}"
    assert (outputs >= lb.unsqueeze(0) - tol).all(), f"LB violated: {(lb.unsqueeze(0) - outputs).max():.6f}"


# ---- Tests ------------------------------------------------------------------

def test_bert_attention_4d_bilinear_sound():
    """Batched (4D) attention matmul through zonotope verification."""
    torch.manual_seed(400)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()

    attn = BertAttention(hidden=16, num_heads=2, head_dim=8).eval()
    center = torch.randn(3, 16) * 0.1; eps = 0.01
    onnx_model = onnx_export(attn, ([3, 16],))
    op = zono.interpret(onnx_model)
    ub, lb = op(_make_input(center, eps)).ublb()
    assert ub.shape == torch.Size([3, 16])
    with torch.no_grad():
        outputs = torch.stack([attn(s) for s in _sample_inputs(center, eps)])
    _check_bounds(outputs, ub, lb)


def test_bert_single_block_sound():
    """Single BERT block (attention + FFN + residuals) soundness."""
    torch.manual_seed(401)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()

    block = BertBlock(hidden=16, intermediate=32, num_heads=2, head_dim=8).eval()
    center = torch.randn(3, 16) * 0.1; eps = 0.005
    onnx_model = onnx_export(block, ([3, 16],))
    op = zono.interpret(onnx_model)
    ub, lb = op(_make_input(center, eps)).ublb()
    assert ub.shape == torch.Size([3, 16])
    with torch.no_grad():
        outputs = torch.stack([block(s) for s in _sample_inputs(center, eps, 1000)])
    _check_bounds(outputs, ub, lb, tol=1e-3)


def test_bert_full_model_sound():
    """Full BERT model (1 layer) end-to-end soundness."""
    torch.manual_seed(402)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()

    model = BertModel(hidden=16, intermediate=32, num_heads=2, head_dim=8,
                      num_layers=1, num_classes=2).eval()
    center = torch.randn(3, 16) * 0.1; eps = 0.005
    onnx_model = onnx_export(model, ([3, 16],))
    op = zono.interpret(onnx_model)
    ub, lb = op(_make_input(center, eps)).ublb()
    assert ub.shape == torch.Size([2])
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in _sample_inputs(center, eps, 1000)])
    _check_bounds(outputs, ub, lb, tol=1e-3)


def test_bert_differential_sound():
    """Differential verification: original vs weight-perturbed BERT."""
    torch.manual_seed(403)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()

    from boundlab.diff.net import diff_net
    from boundlab.diff.expr import DiffExpr3
    from boundlab.diff.zono3 import interpret as diff_interpret

    model1 = BertModel(hidden=16, intermediate=32, num_heads=2, head_dim=8,
                       num_layers=1, num_classes=2).eval()
    model2 = copy.deepcopy(model1)
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * 0.005)
    model2.eval()

    center = torch.randn(3, 16) * 0.1; eps = 0.005
    merged = diff_net(onnx_export(model1, ([3, 16],)), onnx_export(model2, ([3, 16],)))
    op = diff_interpret(merged)
    x_expr = _make_input(center, eps)
    out = op(DiffExpr3(x_expr, x_expr, expr.ConstVal(torch.zeros_like(center))))
    diff_ub, diff_lb = out.diff.ublb()
    assert diff_ub.shape == torch.Size([2])
    assert (diff_ub >= diff_lb).all()

    for _ in range(500):
        x = center + (torch.rand_like(center) * 2 - 1) * eps
        with torch.no_grad():
            d = model1(x) - model2(x)
        assert (d <= diff_ub + 1e-3).all(), f"UB violated"
        assert (d >= diff_lb - 1e-3).all(), f"LB violated"
