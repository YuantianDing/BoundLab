"""Soundness tests for BoundLab zonotope certification of the DeepT BERT model.

Covers:

* **State-dict loading** — the HuggingFace checkpoint maps correctly to
  BoundLab's ``DeepTBert`` wrapper.
* **Bounds enclosure** — concrete outputs under random L∞ perturbations lie
  inside the certified [lb, ub] interval at every logit, across a wide range
  of ε, sequence lengths, depths, and random seeds.
* **Progressive-cut analysis** — each sub-stage of the model (LayerNormNoVar,
  QKV projections, Q@K^T bilinear, softmax, attn@V, full MHA, TransformerBlock,
  full BERT) is individually verified for soundness.
* **PGD cross-check** — for certified samples, PGD attack at the same ε
  cannot flip the prediction.
* **Corner perturbations** — extreme-point δ ∈ {−ε, +ε}^shape probes.

Run with::

    cd bert_test
    pytest test_bert_soundness.py -v
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export


# ---------------------------------------------------------------------------
# Model definition (matches BoundLab's certify_sst2_bert.py exactly)
# ---------------------------------------------------------------------------

class LayerNormNoVar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
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


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.attn_norm = LayerNormNoVar(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ff_norm = LayerNormNoVar(d_model)
    def forward(self, x):
        x = self.attn_norm(x + self.attn(x))
        x = self.ff_norm(x + self.ff2(self.relu(self.ff1(x))))
        return x


class DeepTBert(nn.Module):
    def __init__(self, d_model=128, d_ff=128, num_heads=4, num_layers=3,
                 num_classes=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads) for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        pooled = torch.tanh(self.pooler(x.mean(dim=0)))
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent

def _load_bert(num_layers: int = 3) -> DeepTBert:
    with open(_HERE / "config.json") as f:
        cfg = json.load(f)
    st = torch.load(_HERE / "pytorch_model.bin", map_location="cpu",
                     weights_only=False)
    m = DeepTBert(cfg["hidden_size"], cfg["intermediate_size"],
                  cfg["num_attention_heads"], num_layers)
    with torch.no_grad():
        for i in range(num_layers):
            b, p = m.blocks[i], f"bert.encoder.layer.{i}"
            for a, k in [("query", "query"), ("key", "key"), ("value", "value")]:
                getattr(b.attn, a).weight.copy_(
                    st[f"{p}.attention.self.{k}.weight"])
                getattr(b.attn, a).bias.copy_(
                    st[f"{p}.attention.self.{k}.bias"])
            b.attn.out_proj.weight.copy_(
                st[f"{p}.attention.output.dense.weight"])
            b.attn.out_proj.bias.copy_(
                st[f"{p}.attention.output.dense.bias"])
            b.attn_norm.weight.copy_(
                st[f"{p}.attention.output.LayerNorm.weight"])
            b.attn_norm.bias.copy_(
                st[f"{p}.attention.output.LayerNorm.bias"])
            b.ff1.weight.copy_(st[f"{p}.intermediate.dense.weight"])
            b.ff1.bias.copy_(st[f"{p}.intermediate.dense.bias"])
            b.ff2.weight.copy_(st[f"{p}.output.dense.weight"])
            b.ff2.bias.copy_(st[f"{p}.output.dense.bias"])
            b.ff_norm.weight.copy_(st[f"{p}.output.LayerNorm.weight"])
            b.ff_norm.bias.copy_(st[f"{p}.output.LayerNorm.bias"])
        m.pooler.weight.copy_(st["bert.pooler.dense.weight"])
        m.pooler.bias.copy_(st["bert.pooler.dense.bias"])
        m.classifier.weight.copy_(st["classifier.weight"])
        m.classifier.bias.copy_(st["classifier.bias"])
    return m.eval()


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
FP_ATOL = 1e-4
N_PERTURBATIONS = 200


# ---------------------------------------------------------------------------
# Soundness helper
# ---------------------------------------------------------------------------

def _enclosure_check(model, inp_shape, inp, eps, n_pert=N_PERTURBATIONS):
    """Returns (center_viol, pert_viol, width) tuple."""
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()
    gm = onnx_export(model, (list(inp_shape),))
    op = zono.interpret(gm)
    with torch.no_grad():
        y = model(inp)
    ub, lb = op(
        expr.ConstVal(inp) + eps * expr.LpEpsilon(list(inp_shape))
    ).ublb()
    cu = (y - ub).clamp(min=0).max().item()
    cl = (lb - y).clamp(min=0).max().item()
    g = torch.Generator().manual_seed(0)
    mu, ml = 0.0, 0.0
    for _ in range(n_pert):
        d = (torch.rand(inp.shape, generator=g) * 2 - 1) * eps
        with torch.no_grad():
            yp = model(inp + d)
        mu = max(mu, (yp - ub).clamp(min=0).max().item())
        ml = max(ml, (lb - yp).clamp(min=0).max().item())
    viol = max(cu, cl, mu, ml)
    width = (ub - lb).max().item()
    return viol, width


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_1L():
    return _load_bert(1)


@pytest.fixture(scope="session")
def model_3L():
    return _load_bert(3)


@pytest.fixture(scope="session")
def emb5():
    """Random (5, 128) embeddings."""
    torch.manual_seed(42)
    return torch.randn(5, 128)


# ---------------------------------------------------------------------------
# Tests — Progressive-cut (1-layer, S=5)
# ---------------------------------------------------------------------------

class TestProgressiveCut:
    """Verify soundness at each sub-stage of a single Transformer block."""

    def _block(self, model_1L):
        return model_1L.blocks[0]

    def test_layer_norm(self, model_1L, emb5):
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.n = model_1L.blocks[0].attn_norm
            def forward(s, x):
                return s.n(x)
        v, _ = _enclosure_check(M().eval(), (5, 128), emb5, 1e-3)
        assert v < FP_ATOL

    def test_qkv_projections(self, model_1L, emb5):
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.a = model_1L.blocks[0].attn
            def forward(s, x):
                return s.a.query(x) + s.a.key(x) + s.a.value(x)
        v, _ = _enclosure_check(M().eval(), (5, 128), emb5, 1e-3)
        assert v < FP_ATOL

    def test_qkt_bilinear(self, model_1L, emb5):
        """Q @ K^T — the bilinear matmul stage that was problematic for ViT."""
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.a = model_1L.blocks[0].attn
            def forward(s, x):
                S, _ = x.shape
                h = s.a.num_heads
                Q = s.a.query(x).reshape(S, h, -1).permute(1, 0, 2)
                K = s.a.key(x).reshape(S, h, -1).permute(1, 0, 2)
                return (Q @ K.transpose(-2, -1)) / s.a.scale
        v, _ = _enclosure_check(M().eval(), (5, 128), emb5, 1e-3)
        assert v < FP_ATOL

    def test_softmax(self, model_1L, emb5):
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.a = model_1L.blocks[0].attn
            def forward(s, x):
                S, _ = x.shape
                h = s.a.num_heads
                Q = s.a.query(x).reshape(S, h, -1).permute(1, 0, 2)
                K = s.a.key(x).reshape(S, h, -1).permute(1, 0, 2)
                return ((Q @ K.transpose(-2, -1)) / s.a.scale).softmax(-1)
        v, _ = _enclosure_check(M().eval(), (5, 128), emb5, 1e-3)
        assert v < FP_ATOL

    def test_attn_times_v(self, model_1L, emb5):
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.a = model_1L.blocks[0].attn
            def forward(s, x):
                S, _ = x.shape
                h = s.a.num_heads
                Q = s.a.query(x).reshape(S, h, -1).permute(1, 0, 2)
                K = s.a.key(x).reshape(S, h, -1).permute(1, 0, 2)
                V = s.a.value(x).reshape(S, h, -1).permute(1, 0, 2)
                a = ((Q @ K.transpose(-2, -1)) / s.a.scale).softmax(-1)
                return (a @ V).permute(1, 0, 2).reshape(S, -1)
        v, _ = _enclosure_check(M().eval(), (5, 128), emb5, 1e-3)
        assert v < FP_ATOL

    def test_full_mha(self, model_1L, emb5):
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.a = model_1L.blocks[0].attn
            def forward(s, x):
                return s.a(x)
        v, _ = _enclosure_check(M().eval(), (5, 128), emb5, 1e-3)
        assert v < FP_ATOL

    def test_full_block(self, model_1L, emb5):
        v, _ = _enclosure_check(model_1L.blocks[0], (5, 128), emb5, 1e-3)
        assert v < FP_ATOL

    def test_full_1layer_bert(self, model_1L, emb5):
        v, _ = _enclosure_check(model_1L, (5, 128), emb5, 1e-3)
        assert v < FP_ATOL


# ---------------------------------------------------------------------------
# Tests — ε sweep
# ---------------------------------------------------------------------------

class TestEpsilonSweep:
    @pytest.mark.parametrize("eps", [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1])
    def test_1layer_eps(self, model_1L, emb5, eps):
        v, _ = _enclosure_check(model_1L, (5, 128), emb5, eps)
        assert v < FP_ATOL, f"violation {v:.4f} at eps={eps}"


# ---------------------------------------------------------------------------
# Tests — Sequence length
# ---------------------------------------------------------------------------

class TestSeqLength:
    @pytest.mark.parametrize("S", [3, 5, 10, 15])
    def test_1layer_seqlen(self, model_1L, S):
        torch.manual_seed(S)
        emb = torch.randn(S, 128)
        v, _ = _enclosure_check(model_1L, (S, 128), emb, 0.01)
        assert v < FP_ATOL, f"violation {v:.4f} at S={S}"


# ---------------------------------------------------------------------------
# Tests — Depth
# ---------------------------------------------------------------------------

class TestDepth:
    @pytest.mark.parametrize("nl", [1, 2, 3])
    def test_depth(self, nl, emb5):
        m = _load_bert(nl)
        v, _ = _enclosure_check(m, (5, 128), emb5, 0.005)
        assert v < FP_ATOL, f"violation {v:.4f} at {nl}-layer"


# ---------------------------------------------------------------------------
# Tests — Multiple seeds
# ---------------------------------------------------------------------------

class TestMultipleSeeds:
    @pytest.mark.parametrize("seed", range(5))
    def test_3layer_seed(self, model_3L, seed):
        torch.manual_seed(seed)
        emb = torch.randn(5, 128)
        v, _ = _enclosure_check(model_3L, (5, 128), emb, 0.01)
        assert v < FP_ATOL, f"violation {v:.4f} at seed={seed}"


# ---------------------------------------------------------------------------
# Tests — Corner perturbations
# ---------------------------------------------------------------------------

class TestCornerPerturbations:
    @pytest.mark.parametrize("eps", [1e-3, 1e-2])
    def test_corners(self, model_1L, emb5, eps):
        """±ε corners are extremal L∞ perturbations."""
        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()
        gm = onnx_export(model_1L, ([5, 128],))
        op = zono.interpret(gm)
        ub, lb = op(
            expr.ConstVal(emb5) + eps * expr.LpEpsilon([5, 128])
        ).ublb()
        g = torch.Generator().manual_seed(77)
        for k in range(30):
            signs = torch.randint(0, 2, emb5.shape, generator=g).float() * 2 - 1
            delta = signs * eps
            with torch.no_grad():
                yp = model_1L(emb5 + delta)
            assert torch.all(yp >= lb - FP_ATOL), \
                f"corner {k}: lb violation {(lb - yp).max():.3e}"
            assert torch.all(yp <= ub + FP_ATOL), \
                f"corner {k}: ub violation {(yp - ub).max():.3e}"


# ---------------------------------------------------------------------------
# Tests — PGD cross-check
# ---------------------------------------------------------------------------

class TestPGDCrosscheck:
    def test_pgd_cannot_break_certified_3layer(self, model_3L):
        """PGD must fail on certified samples."""
        eps = 0.005
        any_certified = False
        for seed in range(5):
            torch.manual_seed(seed)
            emb = torch.randn(5, 128)
            prop._UB_CACHE.clear()
            prop._LB_CACHE.clear()
            gm = onnx_export(model_3L, ([5, 128],))
            op = zono.interpret(gm)
            with torch.no_grad():
                pred = int(model_3L(emb).argmax().item())
            ub, lb = op(
                expr.ConstVal(emb) + eps * expr.LpEpsilon([5, 128])
            ).ublb()
            ub_o = ub.clone()
            ub_o[pred] = float("-inf")
            certified = float(lb[pred] - ub_o.max()) > 0.0
            if not certified:
                continue
            any_certified = True
            # PGD
            alpha = eps / 4
            for _ in range(5):  # restarts
                delta = (torch.rand_like(emb) * 2 * eps - eps).detach()
                delta.requires_grad_(True)
                for _ in range(50):
                    out = model_3L(emb + delta)
                    mask = torch.ones_like(out, dtype=torch.bool)
                    mask[pred] = False
                    margin = out[pred] - out[mask].max()
                    if int(out.argmax().item()) != pred:
                        pytest.fail(
                            f"SOUNDNESS: PGD flipped certified sample "
                            f"(seed={seed}, eps={eps})"
                        )
                    grad, = torch.autograd.grad(margin, delta)
                    with torch.no_grad():
                        delta.sub_(alpha * grad.sign()).clamp_(-eps, eps)
                    delta.requires_grad_(True)
        if not any_certified:
            pytest.skip("No sample certified; PGD check vacuous.")


# ---------------------------------------------------------------------------
# Tests — Zonotope invariants
# ---------------------------------------------------------------------------

class TestZonotopeInvariants:
    @pytest.mark.parametrize("eps", [0.0, 1e-5, 1e-3, 1e-2])
    def test_lb_le_ub(self, model_1L, emb5, eps):
        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()
        gm = onnx_export(model_1L, ([5, 128],))
        op = zono.interpret(gm)
        ub, lb = op(
            expr.ConstVal(emb5) + eps * expr.LpEpsilon([5, 128])
        ).ublb()
        assert torch.all(lb <= ub + FP_ATOL)

    def test_margin_monotone(self, model_1L, emb5):
        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()
        gm = onnx_export(model_1L, ([5, 128],))
        op = zono.interpret(gm)
        with torch.no_grad():
            pred = int(model_1L(emb5).argmax().item())
        margins = []
        # Skip eps=0: zonotope uses a different code path (concrete eval) at
        # eps=0, so the margin can jump when switching to linearizer path.
        for eps in [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]:
            prop._UB_CACHE.clear()
            prop._LB_CACHE.clear()
            ub, lb = op(
                expr.ConstVal(emb5) + eps * expr.LpEpsilon([5, 128])
            ).ublb()
            ub_o = ub.clone()
            ub_o[pred] = float("-inf")
            margins.append(float(lb[pred] - ub_o.max()))
        for a, b in zip(margins, margins[1:]):
            assert a + FP_ATOL >= b, f"margin grew: {margins}"
