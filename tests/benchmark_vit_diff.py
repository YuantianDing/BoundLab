"""ViT differential verification benchmark.

Uses the full repo ViT architecture (dim=48, heads=3, dim_head=16, mlp_dim=96)
with LayerNormNoVar. Matches vit_pgd_2_3_16 config but with no_var norm.

Compares three model modification strategies:
1. Weight perturbation (Gaussian noise)
2. Weight rounding (simulate quantization)
3. Weight pruning (zero out small weights)

For each, measures:
- Monte Carlo empirical max difference
- Differential zonotope bound
- Zonotope subtraction bound
- Interval (independent) bound

Usage:
    python tests/benchmark_vit_diff.py --all
    python tests/benchmark_vit_diff.py --eps 0.002 --weight-noise 0.005
    python tests/benchmark_vit_diff.py --round-bits 4
    python tests/benchmark_vit_diff.py --prune-threshold 0.02
    python tests/benchmark_vit_diff.py --layers 2 --eps 0.001 --weight-noise 0.003
"""

import argparse
import copy
import time
import warnings

import torch
from torch import nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

warnings.filterwarnings("ignore")


# ---- ViT components (matches examples/vit/vit.py) --------------------------

class LayerNormNoVar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias


class Attention(nn.Module):
    def __init__(self, dim, heads=3, dim_head=16):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
    def forward(self, x):
        n, _ = x.shape; h = self.heads
        q = self.to_q(x).reshape(n, h, -1).permute(1, 0, 2)
        k = self.to_k(x).reshape(n, h, -1).permute(1, 0, 2)
        v = self.to_v(x).reshape(n, h, -1).permute(1, 0, 2)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).permute(1, 0, 2).reshape(n, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attn_norm = LayerNormNoVar(dim)
        self.attn = Attention(dim, heads, dim_head)
        self.ff_norm = LayerNormNoVar(dim)
        self.ff = FeedForward(dim, mlp_dim)
    def forward(self, x):
        x = self.attn(self.attn_norm(x)) + x
        x = self.ff(self.ff_norm(x)) + x
        return x


class ViT(nn.Module):
    """Full ViT matching repo config. No batch dim: (C, H, W) -> (num_classes,)."""
    def __init__(self, image_size=32, patch_size=16, num_classes=10,
                 dim=48, depth=1, heads=3, mlp_dim=96, channels=3, dim_head=16):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_conv = nn.Conv2d(channels, dim, kernel_size=patch_size,
                                    stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(dim))
        self.pos_embedding = nn.Parameter(torch.zeros(num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim)
            for _ in range(depth)
        ])
        self.head_norm = LayerNormNoVar(dim)
        self.head_linear = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.patch_conv(img.unsqueeze(0))
        x = x.flatten(2).squeeze(0).transpose(0, 1)
        cls = self.cls_token.unsqueeze(0)
        x = torch.cat([cls, x], dim=0)
        x = x + self.pos_embedding[:x.shape[0]]
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=0)
        return self.head_linear(self.head_norm(x))


# ---- Model modification strategies -----------------------------------------

def perturb_weights(model, noise_scale):
    """Add Gaussian noise to all parameters."""
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * noise_scale)
    return model2


def round_weights(model, bits):
    """Round weights to simulate fixed-point quantization."""
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            abs_max = p.abs().max()
            if abs_max == 0:
                continue
            scale = (2 ** (bits - 1) - 1) / abs_max
            p.copy_(torch.round(p * scale) / scale)
    return model2


def prune_weights(model, threshold):
    """Zero out weights below threshold (magnitude pruning)."""
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            mask = p.abs() < threshold
            p[mask] = 0.0
    return model2


# ---- Verification methods --------------------------------------------------

def monte_carlo(model1, model2, center, eps, n_samples=5000):
    """Empirical max |f(x) - g(x)| over random samples."""
    samples = center.unsqueeze(0) + eps * (torch.rand(n_samples, *center.shape) * 2 - 1)
    max_diff = 0.0
    with torch.no_grad():
        for s in samples:
            d = (model1(s) - model2(s)).abs().max().item()
            if d > max_diff:
                max_diff = d
    return max_diff


def differential_verify(model1, model2, center, eps):
    """Differential zonotope verification."""
    from boundlab.diff.net import diff_net
    from boundlab.diff.expr import DiffExpr3
    from boundlab.diff.zono3 import interpret as diff_interpret

    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    onnx1 = onnx_export(model1, ([3, 32, 32],))
    onnx2 = onnx_export(model2, ([3, 32, 32],))
    merged = diff_net(onnx1, onnx2)
    op = diff_interpret(merged)

    x_expr = expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps
    triple = DiffExpr3(x_expr, x_expr, expr.ConstVal(torch.zeros_like(center)))
    out = op(triple)
    diff_ub, diff_lb = out.diff.ublb()
    return max(diff_ub.abs().max().item(), diff_lb.abs().max().item())


def interval_verify(model1, model2, center, eps):
    """Independent zonotope verification (no correlation tracking)."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    onnx1 = onnx_export(model1, ([3, 32, 32],))
    onnx2 = onnx_export(model2, ([3, 32, 32],))
    op1 = zono.interpret(onnx1)
    op2 = zono.interpret(onnx2)

    x_expr = expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps
    y1 = op1(x_expr)
    y2 = op2(x_expr)
    ub1, lb1 = y1.ublb()
    ub2, lb2 = y2.ublb()
    return max((ub1 - lb2).max().item(), (ub2 - lb1).max().item())


def zonotope_sub_verify(model1, model2, center, eps):
    """Zonotope subtraction (shared noise symbols, no diff tracking)."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    onnx1 = onnx_export(model1, ([3, 32, 32],))
    onnx2 = onnx_export(model2, ([3, 32, 32],))
    op1 = zono.interpret(onnx1)
    op2 = zono.interpret(onnx2)

    x_expr = expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps
    y1 = op1(x_expr)
    y2 = op2(x_expr)
    diff = y1 - y2
    ub, lb = diff.ublb()
    return max(ub.abs().max().item(), lb.abs().max().item())


# ---- Main ------------------------------------------------------------------

def run_benchmark(model1, model2, center, eps, label, methods_to_run=None):
    """Run verification methods and print results."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  eps={eps}")
    print(f"{'='*65}")

    total_diff = 0.0
    total_params = 0
    max_param_diff = 0.0
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            d = (p1 - p2).abs()
            total_diff += d.sum().item()
            total_params += p1.numel()
            max_param_diff = max(max_param_diff, d.max().item())
    print(f"  Weight L1 diff:      {total_diff:.6f} ({total_diff/total_params:.6f} per param)")
    print(f"  Weight Linf diff:    {max_param_diff:.6f}")
    print(f"  Total params:        {total_params}")
    print()

    all_methods = [
        ("Monte Carlo (5000 samples)", monte_carlo),
        ("differential",               differential_verify),
        ("zonotope_sub",                zonotope_sub_verify),
        ("interval",                    interval_verify),
    ]

    if methods_to_run is not None:
        all_methods = [(n, f) for n, f in all_methods if n.startswith("Monte") or n in methods_to_run]

    mc_bound = None
    for name, fn in all_methods:
        t0 = time.time()
        try:
            bound = fn(model1, model2, center, eps) if name.startswith("Monte") else fn(model1, model2, center, eps)
            elapsed = time.time() - t0
            if mc_bound is None:
                mc_bound = bound
            ratio = f"{bound/mc_bound:.1f}x MC" if mc_bound and mc_bound > 0 else "N/A"
            print(f"  {name + ':':<40} {bound:.6f}  ({ratio}, {elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            msg = str(e)
            print(f"  {name + ':':<40} FAILED ({elapsed:.1f}s)")
            print(f"    {msg[:100]}{'...' if len(msg) > 100 else ''}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ViT differential verification benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.002,
                        help="L-inf input perturbation radius")

    # Architecture
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of transformer layers (default: 1)")
    parser.add_argument("--dim", type=int, default=48,
                        help="Embedding dimension (default: 48, matching repo)")
    parser.add_argument("--heads", type=int, default=3,
                        help="Number of attention heads (default: 3)")
    parser.add_argument("--dim-head", type=int, default=16,
                        help="Dimension per head (default: 16)")
    parser.add_argument("--mlp-dim", type=int, default=96,
                        help="FFN hidden dimension (default: 96)")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Patch size (default: 16, 4 patches)")

    # Model modification
    parser.add_argument("--weight-noise", type=float, default=None,
                        help="Gaussian noise scale for weight perturbation")
    parser.add_argument("--round-bits", type=int, default=None,
                        help="Number of bits for weight rounding (e.g. 8, 4)")
    parser.add_argument("--prune-threshold", type=float, default=None,
                        help="Magnitude threshold for weight pruning")
    parser.add_argument("--all", action="store_true",
                        help="Run all three strategies with default params")

    # Methods
    parser.add_argument("--skip-interval", action="store_true",
                        help="Skip interval verification (fastest)")
    parser.add_argument("--skip-zono-sub", action="store_true",
                        help="Skip zonotope subtraction")
    parser.add_argument("--skip-diff", action="store_true",
                        help="Skip differential verification")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    num_patches = (32 // args.patch_size) ** 2
    num_tokens = num_patches + 1

    model1 = ViT(
        image_size=32, patch_size=args.patch_size, num_classes=10,
        dim=args.dim, depth=args.layers, heads=args.heads,
        mlp_dim=args.mlp_dim, dim_head=args.dim_head,
    ).eval()
    center = torch.randn(3, 32, 32) * 0.05

    n_params = sum(p.numel() for p in model1.parameters())
    print(f"ViT config: dim={args.dim}, heads={args.heads}, dim_head={args.dim_head}, "
          f"mlp_dim={args.mlp_dim}, layers={args.layers}")
    print(f"Patch size: {args.patch_size}, tokens: {num_patches}+CLS = {num_tokens}")
    print(f"Parameters: {n_params:,}")
    print(f"Input: (3, 32, 32), eps={args.eps}")

    methods = []
    if not args.skip_diff:
        methods.append("differential")
    if not args.skip_zono_sub:
        methods.append("zonotope_sub")
    if not args.skip_interval:
        methods.append("interval")

    ran_any = False

    if args.weight_noise is not None or args.all:
        noise = args.weight_noise or 0.005
        model2 = perturb_weights(model1, noise)
        run_benchmark(model1, model2, center, args.eps,
                      f"Weight perturbation (noise={noise})", methods)
        ran_any = True

    if args.round_bits is not None or args.all:
        bits = args.round_bits or 8
        model2 = round_weights(model1, bits)
        run_benchmark(model1, model2, center, args.eps,
                      f"Weight rounding ({bits}-bit quantization)", methods)
        ran_any = True

    if args.prune_threshold is not None or args.all:
        threshold = args.prune_threshold or 0.01
        model2 = prune_weights(model1, threshold)
        n_pruned = sum((p.abs() < threshold).sum().item() for p in model1.parameters())
        n_total = sum(p.numel() for p in model1.parameters())
        pct = 100 * n_pruned / n_total
        run_benchmark(model1, model2, center, args.eps,
                      f"Weight pruning (threshold={threshold}, {n_pruned}/{n_total}={pct:.0f}% pruned)",
                      methods)
        ran_any = True

    if not ran_any:
        model2 = perturb_weights(model1, 0.005)
        run_benchmark(model1, model2, center, args.eps,
                      "Weight perturbation (noise=0.005)", methods)


if __name__ == "__main__":
    main()
