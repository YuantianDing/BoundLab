"""Differential verification benchmark for BERT-smaller-3.

Compares three approaches to bounding f₁(x) − f₂(x) where f₁ is the original
model and f₂ has perturbed weights:

1. **Differential verification** (DiffExpr3): tracks the difference symbolically
   through the network, sharing epsilon variables between the two branches.
   Expected to give the tightest bounds.

2. **Interval subtraction** (baseline): run standard zonotope verification on
   each model independently, compute [lb₁ - ub₂, ub₁ - lb₂]. This is the
   loosest baseline — no correlation between the two runs.

3. **Zonotope subtraction** (shared-input baseline): run standard zonotope
   verification on each model with the *same* input expression (shared epsilon),
   then subtract the output expressions and compute bounds on the difference.
   Tighter than interval subtraction because shared epsilons partially cancel,
   but looser than differential verification because the diff isn't tracked
   through nonlinearities.

Usage:
    python tests/benchmark_diff_verification.py [--layers 1] [--seq-len 4] [--eps 0.01] [--weight-noise 0.01]
"""

import argparse
import math
import time
import warnings

import torch
from torch import nn

# Suppress noisy LinearOp fusion/jacobian warnings by default
warnings.filterwarnings("ignore", message=".*LinearOp.*")
warnings.filterwarnings("ignore", message=".*norm_input.*")
warnings.filterwarnings("ignore", message=".*jacobian.*")
warnings.filterwarnings("ignore", message=".*fuse.*")

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export
from boundlab.diff.net import diff_net
from boundlab.diff.zono3.expr import DiffExpr3
from boundlab.diff.zono3 import interpret as diff_interpret

class LayerNormNoVar(nn.Module):
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
        context = torch.matmul(attn, V)
        context = context.permute(1, 0, 2).reshape(S, -1)
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
    def __init__(self, hidden=64, intermediate=128, num_heads=4, head_dim=16,
                 num_layers=3, num_classes=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            BERTBlock(hidden, intermediate, num_heads, head_dim)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        cls_token = x[0]
        return self.classifier(cls_token)


def perturb_weights(model, noise_scale):
    """Return a copy of model with Gaussian noise added to all parameters."""
    import copy
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * noise_scale)
    return model2


def make_input_expr(center, eps):
    """Build ConstVal(center) + eps * LpEpsilon(shape)."""
    return expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps


def sample_concrete_diffs(model1, model2, center, eps, n_samples=5000):
    """Monte Carlo: sample f₁(x) - f₂(x) for x in L∞ ball."""
    diffs = []
    with torch.no_grad():
        for _ in range(n_samples):
            noise = (torch.rand_like(center) * 2 - 1) * eps
            x = center + noise
            diffs.append(model1(x) - model2(x))
    return torch.stack(diffs)

def differential_verification(model1, model2, center, eps, input_shape):
    """Run differential verification using DiffExpr3 + diff_net.

    Returns (diff_ub, diff_lb, elapsed_seconds).
    """
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    onnx1 = onnx_export(model1, (input_shape,))
    onnx2 = onnx_export(model2, (input_shape,))
    merged = diff_net(onnx1, onnx2)

    op = diff_interpret(merged)

    x_expr = make_input_expr(center, eps)
    # For differential: same input to both networks
    triple = DiffExpr3(x_expr, x_expr, expr.ConstVal(torch.zeros_like(center)))

    t0 = time.perf_counter()
    out = op(triple)
    diff_ub, diff_lb = out.diff.ublb()
    elapsed = time.perf_counter() - t0

    return diff_ub, diff_lb, elapsed

def interval_subtraction(model1, model2, center, eps, input_shape):
    """Run standard zonotope on each model independently, then subtract intervals.

    diff ∈ [lb₁ - ub₂, ub₁ - lb₂]

    Returns (diff_ub, diff_lb, elapsed_seconds).
    """
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    onnx1 = onnx_export(model1, (input_shape,))
    onnx2 = onnx_export(model2, (input_shape,))

    t0 = time.perf_counter()

    # Model 1
    op1 = zono.interpret(onnx1)
    x1 = make_input_expr(center, eps)
    y1 = op1(x1)
    ub1, lb1 = y1.ublb()

    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    # Model 2
    op2 = zono.interpret(onnx2)
    x2 = make_input_expr(center, eps)
    y2 = op2(x2)
    ub2, lb2 = y2.ublb()

    # Interval subtraction: [lb1 - ub2, ub1 - lb2]
    diff_ub = ub1 - lb2
    diff_lb = lb1 - ub2

    elapsed = time.perf_counter() - t0
    return diff_ub, diff_lb, elapsed

def zonotope_subtraction(model1, model2, center, eps, input_shape):
    """Run standard zonotope on each model with *shared* input epsilon,
    then compute bounds on (y1 - y2).

    Returns (diff_ub, diff_lb, elapsed_seconds).
    """
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    onnx1 = onnx_export(model1, (input_shape,))
    onnx2 = onnx_export(model2, (input_shape,))

    t0 = time.perf_counter()

    # Same input expression for both — shared epsilon
    x_shared = make_input_expr(center, eps)

    op1 = zono.interpret(onnx1)
    y1 = op1(x_shared)

    op2 = zono.interpret(onnx2)
    y2 = op2(x_shared)

    # Symbolic subtraction — shared epsilons partially cancel
    diff_expr = y1 - y2
    diff_ub, diff_lb = diff_expr.ublb()

    elapsed = time.perf_counter() - t0
    return diff_ub, diff_lb, elapsed

def run_benchmark(num_layers, seq_len, eps, weight_noise, hidden=64, intermediate=128, num_heads=4, head_dim=16, seed=42):
    torch.manual_seed(seed)

    input_shape = [seq_len, hidden]

    print(f"\n{'='*70}")
    print(f"Differential Verification Benchmark")
    print(f"  layers={num_layers}, seq_len={seq_len}, hidden={hidden}, intermediate={intermediate}")
    print(f"  heads={num_heads}, head_dim={head_dim}, eps={eps}, weight_noise={weight_noise}")
    print(f"{'='*70}\n")

    # Build models
    model1 = BERTSmaller3(hidden, intermediate, num_heads, head_dim,
                          num_layers=num_layers)
    model1.eval()
    model2 = perturb_weights(model1, weight_noise)
    model2.eval()

    center = torch.randn(seq_len, hidden) * 0.05

    # Monte Carlo ground truth
    print("Sampling ground truth (Monte Carlo)...")
    mc_diffs = sample_concrete_diffs(model1, model2, center, eps, n_samples=5000)
    mc_ub = mc_diffs.max(dim=0).values
    mc_lb = mc_diffs.min(dim=0).values
    mc_width = (mc_ub - mc_lb).sum().item()
    print(f"  MC diff width (sum): {mc_width:.6f}")
    print(f"  MC diff range: [{mc_lb.min():.6f}, {mc_ub.max():.6f}]")

    results = {}

    # Method 1: Differential verification
    print("\nRunning differential verification...")
    try:
        d_ub, d_lb, d_time = differential_verification(
            model1, model2, center, eps, input_shape)
        d_width = (d_ub - d_lb).sum().item()
        results['differential'] = (d_ub, d_lb, d_time, d_width)
        print(f"  Bound width (sum): {d_width:.6f}")
        print(f"  Time: {d_time:.2f}s")
        # Soundness check
        sound = (mc_diffs <= d_ub.unsqueeze(0) + 1e-4).all() and \
                (mc_diffs >= d_lb.unsqueeze(0) - 1e-4).all()
        print(f"  Sound: {sound}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['differential'] = None

    # Method 2: Interval subtraction
    print("\nRunning interval subtraction (independent)...")
    try:
        i_ub, i_lb, i_time = interval_subtraction(
            model1, model2, center, eps, input_shape)
        i_width = (i_ub - i_lb).sum().item()
        results['interval'] = (i_ub, i_lb, i_time, i_width)
        print(f"  Bound width (sum): {i_width:.6f}")
        print(f"  Time: {i_time:.2f}s")
        sound = (mc_diffs <= i_ub.unsqueeze(0) + 1e-4).all() and \
                (mc_diffs >= i_lb.unsqueeze(0) - 1e-4).all()
        print(f"  Sound: {sound}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['interval'] = None

    # Method 3: Zonotope subtraction
    print("\nRunning zonotope subtraction (shared input)...")
    try:
        z_ub, z_lb, z_time = zonotope_subtraction(
            model1, model2, center, eps, input_shape)
        z_width = (z_ub - z_lb).sum().item()
        results['zonotope_sub'] = (z_ub, z_lb, z_time, z_width)
        print(f"  Bound width (sum): {z_width:.6f}")
        print(f"  Time: {z_time:.2f}s")
        sound = (mc_diffs <= z_ub.unsqueeze(0) + 1e-4).all() and \
                (mc_diffs >= z_lb.unsqueeze(0) - 1e-4).all()
        print(f"  Sound: {sound}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['zonotope_sub'] = None

    # Summary
    print(f"\n{'='*70}")
    print("Summary — bound width (sum over output dims, lower is tighter):")
    print(f"{'='*70}")
    print(f"  {'Monte Carlo (empirical):':<35} {mc_width:.6f}")
    for name, res in results.items():
        if res is not None:
            _, _, t, w = res
            ratio = w / mc_width if mc_width > 0 else float('inf')
            print(f"  {name + ':':<35} {w:.6f}  ({ratio:.1f}x MC, {t:.2f}s)")
        else:
            print(f"  {name + ':':<35} FAILED")

    print()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential verification benchmark")
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of BERT layers (default: 1)")
    parser.add_argument("--seq-len", type=int, default=4,
                        help="Sequence length (default: 4)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden/embedding size (default: 64)")
    parser.add_argument("--intermediate", type=int, default=64,
                        help="FFN intermediate size (default: 64, paper small model)")
    parser.add_argument("--heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--eps", type=float, default=0.01,
                        help="Input L∞ perturbation radius (default: 0.01)")
    parser.add_argument("--weight-noise", type=float, default=0.01,
                        help="Std of Gaussian noise added to weights (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    head_dim = args.hidden // args.heads
    run_benchmark(
        num_layers=args.layers,
        seq_len=args.seq_len,
        eps=args.eps,
        weight_noise=args.weight_noise,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_heads=args.heads,
        head_dim=head_dim,
        seed=args.seed,
    )