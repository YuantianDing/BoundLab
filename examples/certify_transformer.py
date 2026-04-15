"""Certify the output difference between a transformer and its weight-perturbed copy.

Given:
  - A simple transformer model (single-head attention + FFN)
  - A perturbation magnitude `weight_eps` applied to all weights
  - An input L∞ ball of radius `input_eps` around a center point

This script computes *certified* upper/lower bounds on:

    max_{x ∈ B(c, input_eps)}  ||f₁(x) − f₂(x)||∞

where f₁ is the original model and f₂ is the perturbed model.

Uses BoundLab's differential zonotope interpreter (DiffExpr3) with
diff_net to merge the two models and propagate bounds through all
layers including softmax attention.

Usage:
    python certify_transformer.py
"""

import copy
import math

import torch
import torch.nn as nn

import boundlab.expr as expr
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export

class SimpleTransformer(nn.Module):
    """Single-head attention + FFN block."""

    def __init__(self, d_model: int, d_k: int, d_ff: int, use_softmax: bool = True):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        self.proj = nn.Linear(d_k, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.inv_scale = 1.0 / math.sqrt(d_k)
        self.use_softmax = use_softmax

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = torch.matmul(Q, K.transpose(0, 1)) * self.inv_scale
        if self.use_softmax:
            attn = torch.softmax(scores, dim=-1)
        else:
            attn = scores
        context = torch.matmul(attn, V)
        out = self.proj(context)
        x = x + out                        # residual
        h = torch.relu(self.ff1(x))
        x = x + self.ff2(h)                # residual
        return x

def perturb_model(model: nn.Module, weight_eps: float, seed: int = 42) -> nn.Module:
    """Return a copy of `model` with all weights perturbed by uniform noise in [-weight_eps, weight_eps]."""
    perturbed = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in perturbed.parameters():
            noise = (torch.rand(p.shape, generator=gen) * 2 - 1) * weight_eps
            p.add_(noise)
    return perturbed

def certify(
    model1: nn.Module,
    model2: nn.Module,
    center: torch.Tensor,
    input_eps: float,
) -> dict:
    """Certify bounds on f1(x) - f2(x) for all x in B∞(center, input_eps).

    Returns a dict with:
        - diff_ub, diff_lb: per-element upper/lower bounds on f1(x)-f2(x)
        - max_diff: certified max |f1(x)-f2(x)| (element-wise)
        - total_bound: certified max ||f1(x)-f2(x)||∞
    """
    shape = list(center.shape)

    # Export both models to ONNX IR
    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))

    # Merge into a single graph with diff_pair nodes
    merged = diff_net(gm1, gm2)

    # Build the differential interpreter
    op = diff_interpret(merged)

    # Create zonotope input: center ± input_eps
    x = expr.ConstVal(center) + input_eps * expr.LpEpsilon(shape)

    # Run through interpreter — produces DiffExpr2, which gets
    # promoted to DiffExpr3 at the first nonlinear diff layer
    out = op(x)

    # Extract diff bounds
    if isinstance(out, DiffExpr3):
        d_ub, d_lb = out.diff.ublb()
    else:
        # DiffExpr2: compute diff as x - y
        d = out.x - out.y
        d_ub, d_lb = d.ublb()

    max_abs = torch.maximum(d_ub.abs(), d_lb.abs())

    return {
        "diff_ub": d_ub,
        "diff_lb": d_lb,
        "max_diff": max_abs,
        "total_bound": max_abs.max().item(),
    }


def verify_with_sampling(
    model1: nn.Module,
    model2: nn.Module,
    center: torch.Tensor,
    input_eps: float,
    n_samples: int = 5000,
) -> torch.Tensor:
    """Sample concrete differences to empirically check the bounds."""
    noise = (torch.rand(n_samples, *center.shape) * 2 - 1) * input_eps
    samples = center.unsqueeze(0) + noise
    with torch.no_grad():
        diffs = torch.stack([model1(s) - model2(s) for s in samples])
    return diffs


def certify_baseline(
    model1: nn.Module,
    model2: nn.Module,
    center: torch.Tensor,
    input_eps: float,
) -> dict:
    """Baseline: certify each model independently, subtract output intervals.

    Computes [lb1, ub1] for f1 and [lb2, ub2] for f2 using independent
    zonotopes (no shared noise), then bounds f1(x)-f2(x) ∈ [lb1-ub2, ub1-lb2].
    This is the loosest approach — it treats the two models as unrelated.
    """
    shape = list(center.shape)

    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))

    # Independent zonotopes — different epsilon symbols
    x1 = expr.ConstVal(center) + input_eps * expr.LpEpsilon(shape)
    x2 = expr.ConstVal(center) + input_eps * expr.LpEpsilon(shape)

    op1 = zono.interpret(gm1)
    op2 = zono.interpret(gm2)
    out1 = op1(x1)
    out2 = op2(x2)

    ub1, lb1 = out1.ublb()
    ub2, lb2 = out2.ublb()

    # Interval subtraction: [lb1 - ub2, ub1 - lb2]
    d_ub = ub1 - lb2
    d_lb = lb1 - ub2
    max_abs = torch.maximum(d_ub.abs(), d_lb.abs())

    return {
        "diff_ub": d_ub,
        "diff_lb": d_lb,
        "max_diff": max_abs,
        "total_bound": max_abs.max().item(),
    }


def certify_zonotope_subtraction(
    model1: nn.Module,
    model2: nn.Module,
    center: torch.Tensor,
    input_eps: float,
) -> dict:
    """Zonotope subtraction: shared input noise, subtract output zonotopes.

    Both models receive the SAME input zonotope (shared epsilon symbols).
    The output expressions are subtracted as Expr objects, so shared noise
    terms partially cancel before bound computation. Tighter than interval
    subtraction, but doesn't track the diff through nonlinearities.
    """
    shape = list(center.shape)

    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))

    # Same zonotope — shared epsilon symbol
    x = expr.ConstVal(center) + input_eps * expr.LpEpsilon(shape)

    op1 = zono.interpret(gm1)
    op2 = zono.interpret(gm2)
    out1 = op1(x)
    out2 = op2(x)

    # Subtract as zonotope expressions — shared epsilons cancel
    d = out1 - out2
    d_ub, d_lb = d.ublb()
    max_abs = torch.maximum(d_ub.abs(), d_lb.abs())

    return {
        "diff_ub": d_ub,
        "diff_lb": d_lb,
        "max_diff": max_abs,
        "total_bound": max_abs.max().item(),
    }


def main():
    torch.manual_seed(0)

    seq_len = 8
    d_model = 32
    d_k = 16
    d_ff = 64
    weight_eps = 0.005
    input_eps = 0.02

    model1 = SimpleTransformer(d_model, d_k, d_ff, use_softmax=True)
    model1.eval()
    model2 = perturb_model(model1, weight_eps)
    model2.eval()

    center = torch.randn(seq_len, d_model) * 0.3

    print("=" * 60)
    print("DIFFERENTIAL VERIFICATION OF TRANSFORMER")
    print("=" * 60)
    print(f"\nModel: SimpleTransformer")
    print(f"  seq_len={seq_len}, d_model={d_model}, d_k={d_k}, d_ff={d_ff}")
    print(f"  Attention: softmax")
    print(f"  FFN activation: ReLU")
    print(f"  Residual connections: yes")
    n_params = sum(p.numel() for p in model1.parameters())
    print(f"  Parameters: {n_params}")
    print(f"\nPerturbation:")
    print(f"  Weight perturbation (ε_w): ±{weight_eps}")
    print(f"  Input perturbation  (ε_x): ±{input_eps}")
    print(f"  Input shape: {list(center.shape)}")

    # Ground truth (Monte Carlo)
    n_samples = 10000
    print(f"\n{'='*60}")
    print(f"GROUND TRUTH (Monte Carlo, {n_samples} samples)")
    print(f"  Sampling uniform x ∈ B∞(center, {input_eps})")
    print(f"  Computing f₁(x) − f₂(x) for each sample")
    diffs = verify_with_sampling(model1, model2, center, input_eps, n_samples)
    empirical_max = diffs.abs().max().item()
    print(f"  Max ||f₁(x) − f₂(x)||∞ = {empirical_max:.6f}")
    print(f"  (lower bound on true max — not a certified bound)")

    # Interval subtraction
    print(f"\n{'='*60}")
    print("METHOD 1: Interval Subtraction")
    print("  Certify f₁ and f₂ independently (no shared noise)")
    print("  Subtract output intervals: [lb₁−ub₂, ub₁−lb₂]")
    r1 = certify_baseline(model1, model2, center, input_eps)
    s1 = (diffs <= r1["diff_ub"].unsqueeze(0) + 1e-5).all() and \
         (diffs >= r1["diff_lb"].unsqueeze(0) - 1e-5).all()
    print(f"  Certified ||f₁(x) − f₂(x)||∞ ≤ {r1['total_bound']:.6f}")
    print(f"  Sound: {s1.item()}")

    # Zonotope subtraction
    print(f"\n{'='*60}")
    print("METHOD 2: Zonotope Subtraction")
    print("  Same input zonotope for both models (shared ε)")
    print("  Subtract output Expr objects, then compute bounds")
    r2 = certify_zonotope_subtraction(model1, model2, center, input_eps)
    s2 = (diffs <= r2["diff_ub"].unsqueeze(0) + 1e-5).all() and \
         (diffs >= r2["diff_lb"].unsqueeze(0) - 1e-5).all()
    print(f"  Certified ||f₁(x) − f₂(x)||∞ ≤ {r2['total_bound']:.6f}")
    print(f"  Sound: {s2.item()}")

    # Differential (DiffExpr3)
    print(f"\n{'='*60}")
    print("METHOD 3: Differential Verification (DiffExpr3)")
    print("  Merge models with diff_net, propagate triple-zonotope")
    print("  Diff component tracked through all layers including softmax")
    r3 = certify(model1, model2, center, input_eps)
    s3 = (diffs <= r3["diff_ub"].unsqueeze(0) + 1e-5).all() and \
         (diffs >= r3["diff_lb"].unsqueeze(0) - 1e-5).all()
    print(f"  Certified ||f₁(x) − f₂(x)||∞ ≤ {r3['total_bound']:.6f}")
    print(f"  Sound: {s3.item()}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Bound':>10} {'Tightness':>10} {'Sound':>6}")
    print(f"{'-'*60}")
    print(f"{'Ground truth (MC)' :<30} {empirical_max:>10.6f} {'—':>10} {'—':>6}")
    for name, r, s in [
        ("Interval subtraction", r1, s1),
        ("Zonotope subtraction", r2, s2),
        ("Differential (ours)", r3, s3),
    ]:
        tight = f"{empirical_max / r['total_bound']:.1%}"
        print(f"{name:<30} {r['total_bound']:>10.6f} {tight:>10} {'✓' if s else '✗':>6}")

    w1 = (r1["diff_ub"] - r1["diff_lb"]).mean().item()
    w2 = (r2["diff_ub"] - r2["diff_lb"]).mean().item()
    w3 = (r3["diff_ub"] - r3["diff_lb"]).mean().item()
    print(f"\nMean interval width:")
    print(f"  Interval subtraction: {w1:.6f}")
    print(f"  Zonotope subtraction: {w2:.6f}  ({(1 - w2/w1)*100:.1f}% reduction vs interval)")
    print(f"  Differential (ours):  {w3:.6f}  ({(1 - w3/w1)*100:.1f}% reduction vs interval)")


if __name__ == "__main__":
    main()