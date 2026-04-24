"""BERT differential verification benchmark.

Matches DeepT paper small model config: hidden=64, num_heads=4, head_dim=16.

Usage:
    python tests/benchmark_bert_diff.py --all
    python tests/benchmark_bert_diff.py --eps 0.01 --weight-noise 0.01
    python tests/benchmark_bert_diff.py --round-bits 8 --eps 0.01
    python tests/benchmark_bert_diff.py --layers 1 --intermediate 64 --eps 0.01 --weight-noise 0.005
"""

import argparse
import copy
import time
import warnings

import torch
from torch import nn
import math

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

warnings.filterwarnings("ignore")


# ---- BERT components --------------------------------------------------------

class LayerNormNoVar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias


class BertAttention(nn.Module):
    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads = num_heads
        self.scale = math.sqrt(head_dim)
        self.query = nn.Linear(hidden, inner)
        self.key = nn.Linear(hidden, inner)
        self.value = nn.Linear(hidden, inner)
        self.out_proj = nn.Linear(inner, hidden)
    def forward(self, x):
        S, _ = x.shape; h = self.num_heads
        Q = self.query(x).reshape(S, h, -1).permute(1, 0, 2)
        K = self.key(x).reshape(S, h, -1).permute(1, 0, 2)
        V = self.value(x).reshape(S, h, -1).permute(1, 0, 2)
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        attn = scores.softmax(dim=-1)
        context = (attn @ V).permute(1, 0, 2).reshape(S, -1)
        return self.out_proj(context)


class BertBlock(nn.Module):
    def __init__(self, hidden, intermediate, num_heads, head_dim):
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
                 num_layers=1, seq_len=5, num_classes=2):
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
        return self.classifier(self.classifier_norm(x[0]))


# ---- Model modifications ---------------------------------------------------

def perturb_weights(model, noise_scale):
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * noise_scale)
    return model2

def round_weights(model, bits):
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            abs_max = p.abs().max()
            if abs_max == 0: continue
            scale = (2 ** (bits - 1) - 1) / abs_max
            p.copy_(torch.round(p * scale) / scale)
    return model2

def prune_weights(model, threshold):
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            p[p.abs() < threshold] = 0.0
    return model2


# ---- Verification methods --------------------------------------------------

def monte_carlo(model1, model2, center, eps, n_samples=5000):
    samples = center.unsqueeze(0) + eps * (torch.rand(n_samples, *center.shape) * 2 - 1)
    max_diff = 0.0
    with torch.no_grad():
        for s in samples:
            d = (model1(s) - model2(s)).abs().max().item()
            max_diff = max(max_diff, d)
    return max_diff

def differential_verify(model1, model2, center, eps, input_shape):
    from boundlab.diff.net import diff_net
    from boundlab.diff.zono3.expr import DiffExpr3
    from boundlab.diff.zono3 import interpret as diff_interpret
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    merged = diff_net(onnx_export(model1, (input_shape,)), onnx_export(model2, (input_shape,)))
    op = diff_interpret(merged)
    x_expr = expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps
    out = op(DiffExpr3(x_expr, x_expr, expr.ConstVal(torch.zeros_like(center))))
    ub, lb = out.diff.ublb()
    return max(ub.abs().max().item(), lb.abs().max().item())

def interval_verify(model1, model2, center, eps, input_shape):
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    op1 = zono.interpret(onnx_export(model1, (input_shape,)))
    op2 = zono.interpret(onnx_export(model2, (input_shape,)))
    x_expr = expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps
    ub1, lb1 = op1(x_expr).ublb()
    ub2, lb2 = op2(x_expr).ublb()
    return max((ub1 - lb2).max().item(), (ub2 - lb1).max().item())

def zonotope_sub_verify(model1, model2, center, eps, input_shape):
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    op1 = zono.interpret(onnx_export(model1, (input_shape,)))
    op2 = zono.interpret(onnx_export(model2, (input_shape,)))
    x_expr = expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps
    diff = op1(x_expr) - op2(x_expr)
    ub, lb = diff.ublb()
    return max(ub.abs().max().item(), lb.abs().max().item())


# ---- Main ------------------------------------------------------------------

def run_benchmark(model1, model2, center, eps, input_shape, label, methods):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  eps={eps}")
    print(f"{'='*65}")
    total_diff = sum((p1 - p2).abs().sum().item() for p1, p2 in zip(model1.parameters(), model2.parameters()))
    n_params = sum(p.numel() for p in model1.parameters())
    print(f"  Weight L1 diff:      {total_diff:.6f} ({total_diff/n_params:.6f} per param)")
    print(f"  Total params:        {n_params}")
    print()

    all_methods = [
        ("Monte Carlo (5000 samples)", lambda: monte_carlo(model1, model2, center, eps)),
        ("differential", lambda: differential_verify(model1, model2, center, eps, input_shape)),
        ("zonotope_sub", lambda: zonotope_sub_verify(model1, model2, center, eps, input_shape)),
        ("interval", lambda: interval_verify(model1, model2, center, eps, input_shape)),
    ]
    if methods:
        all_methods = [(n, f) for n, f in all_methods if n.startswith("Monte") or n in methods]

    mc_bound = None
    for name, fn in all_methods:
        t0 = time.time()
        try:
            bound = fn()
            elapsed = time.time() - t0
            if mc_bound is None: mc_bound = bound
            ratio = f"{bound/mc_bound:.1f}x MC" if mc_bound > 0 else "N/A"
            print(f"  {name + ':':<40} {bound:.6f}  ({ratio}, {elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            msg = str(e)[:100]
            print(f"  {name + ':':<40} FAILED ({elapsed:.1f}s): {msg}")
    print()


def main():
    parser = argparse.ArgumentParser(description="BERT differential verification benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--weight-noise", type=float, default=None)
    parser.add_argument("--round-bits", type=int, default=None)
    parser.add_argument("--prune-threshold", type=float, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip-interval", action="store_true")
    parser.add_argument("--skip-zono-sub", action="store_true")
    parser.add_argument("--skip-diff", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model1 = BertModel(hidden=args.hidden, intermediate=args.intermediate,
                       num_heads=args.num_heads, head_dim=args.head_dim,
                       num_layers=args.layers, num_classes=2).eval()
    center = torch.randn(args.seq_len, args.hidden) * 0.1
    input_shape = [args.seq_len, args.hidden]

    n_params = sum(p.numel() for p in model1.parameters())
    print(f"BERT config: hidden={args.hidden}, intermediate={args.intermediate}, "
          f"heads={args.num_heads}, head_dim={args.head_dim}, layers={args.layers}")
    print(f"Seq length: {args.seq_len}, Parameters: {n_params:,}")
    print(f"Input: ({args.seq_len}, {args.hidden}), eps={args.eps}")

    methods = []
    if not args.skip_diff: methods.append("differential")
    if not args.skip_zono_sub: methods.append("zonotope_sub")
    if not args.skip_interval: methods.append("interval")

    ran_any = False
    if args.weight_noise is not None or args.all:
        noise = args.weight_noise or 0.01
        model2 = perturb_weights(model1, noise)
        run_benchmark(model1, model2, center, args.eps, input_shape,
                      f"Weight perturbation (noise={noise})", methods)
        ran_any = True

    if args.round_bits is not None or args.all:
        bits = args.round_bits or 8
        model2 = round_weights(model1, bits)
        run_benchmark(model1, model2, center, args.eps, input_shape,
                      f"Weight rounding ({bits}-bit quantization)", methods)
        ran_any = True

    if args.prune_threshold is not None or args.all:
        threshold = args.prune_threshold or 0.01
        model2 = prune_weights(model1, threshold)
        n_pruned = sum((p.abs() < threshold).sum().item() for p in model1.parameters())
        pct = 100 * n_pruned / n_params
        run_benchmark(model1, model2, center, args.eps, input_shape,
                      f"Weight pruning (threshold={threshold}, {pct:.0f}% pruned)", methods)
        ran_any = True

    if not ran_any:
        model2 = perturb_weights(model1, 0.01)
        run_benchmark(model1, model2, center, args.eps, input_shape,
                      "Weight perturbation (noise=0.01)", methods)

if __name__ == "__main__":
    main()
