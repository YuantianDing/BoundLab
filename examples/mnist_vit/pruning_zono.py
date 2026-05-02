"""Differential verification of pruned ViT using zonotope methods.

Bounds ||model(x) - model_pruned(x)|| under L∞ perturbation using:
  - zono/interval: Independent interval bounds, subtract
  - zono:          Shared zonotope, subtract (noise cancels)
  - zono3:         Differential zonotope via diff_net (tightest)
  - zono3/gradlin: Differential zonotope with gradient linearization

Compares tightness against Monte Carlo ground truth.

Usage::

    python pruning_zono.py --checkpoint mnist_transformer.pt --eps 0.002 --K 8 --n-samples 5
    python pruning_zono.py --eps 0.004 --K 4 --n-samples 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn, Tensor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff import zono3, zonohex
from boundlab.interp.onnx import onnx_export

from mnist_vit import build_mnist_vit
from certify import PatchifyStage
from certify_pruned import ScoringModel, build_zonotope_no_cat, classify_topk


# ---------------------------------------------------------------------------
# MaskedModel: identical graph structure for both full and pruned
# ---------------------------------------------------------------------------

class MaskedModel(nn.Module):
    """PostConcat with a mask Mul as the first op.

    Both the full model (mask=ones) and pruned model (mask=pruned_zeros)
    use this class, ensuring identical ONNX graph structure for diff_net.
    """

    def __init__(self, vit, mask: Tensor):
        super().__init__()
        attn_block = vit.transformer.layers[0][0]
        ff_block = vit.transformer.layers[0][1]
        self.attn_norm = attn_block.fn.norm
        self.attn = attn_block.fn.fn
        self.heads = self.attn.heads
        self.dim_head = self.attn.dim_head
        self.scale = self.attn.scale
        self.ff_block = ff_block
        self.pool = vit.pool
        self.mlp_head = vit.mlp_head
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.mask
        n, h, d = x.shape[0], self.heads, self.dim_head
        residual = x
        xn = self.attn_norm(x)
        q = self.attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)
        k = self.attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
        v = self.attn.to_v(xn).reshape(n, h, d).permute(1, 0, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_w = scores.softmax(dim=-1)
        out = (attn_w @ v).permute(1, 0, 2).reshape(n, h * d)
        out = self.attn.to_out(out)
        x = residual + out
        x = self.ff_block(x)
        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)


# ---------------------------------------------------------------------------
# Bound-width breakdown helpers
# ---------------------------------------------------------------------------

def _bound_width_reasons(e: expr.Expr) -> dict[str, Tensor]:
    return {
        reason: value.detach()
        for reason, value in e.bound_width_reasons_breakdown().items()
    }


def _merge_reason_breakdowns(
    merged: dict[str, Tensor] | None,
    current: dict[str, Tensor],
) -> dict[str, Tensor]:
    if merged is None:
        return {reason: value.clone() for reason, value in current.items()}
    result = {reason: value.clone() for reason, value in merged.items()}
    zeros = None
    for value in list(result.values()) + list(current.values()):
        zeros = torch.zeros_like(value)
        break
    if zeros is None:
        return result
    for reason in set(result) | set(current):
        lhs = result.get(reason, zeros)
        rhs = current.get(reason, zeros)
        result[reason] = torch.maximum(lhs, rhs)
    return result


def _format_reason_breakdown(breakdown: dict[str, Tensor], width: float) -> str:
    if not breakdown:
        return "(none)"
    parts = []
    for reason, contribution in sorted(
        breakdown.items(),
        key=lambda item: item[1].mean().item(),
        reverse=True,
    ):
        mean = contribution.mean().item()
        pct = 100.0 * mean / width if width > 0 else 0.0
        parts.append(f"{reason}={mean:.4f} ({pct:.1f}%)")
    return ", ".join(parts)


def _mask_cache_key(mask: Tensor) -> bytes:
    """Stable cache key for a binary mask tensor."""
    return mask.detach().to(dtype=torch.uint8).contiguous().cpu().numpy().tobytes()


def _export_masked_onnx(vit, mask: Tensor, cache: dict[bytes, object]):
    """Export and cache a MaskedModel ONNX graph keyed by the binary mask."""
    key = _mask_cache_key(mask)
    if key not in cache:
        cache[key] = onnx_export(MaskedModel(vit, mask).eval(), ([17, 64],))
    return cache[key]


# ---------------------------------------------------------------------------
# Zonotope certification methods
# ---------------------------------------------------------------------------

def certify_int_sub(gm1, gm2, center, eps):
    """Independent interval bounds, then subtract."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    x1 = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
    y1 = zono.interpret(gm1)(x1)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    x2 = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
    y2 = zono.interpret(gm2)(x2)
    d = y1 - y2
    return (*d.ublb(), _bound_width_reasons(d))


def certify_zono_sub(gm1, gm2, center, eps):
    """Shared zonotope, subtract expressions (noise cancels)."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
    op1 = zono.interpret(gm1)
    op2 = zono.interpret(gm2)
    d = op1(x) - op2(x)
    return (*d.ublb(), _bound_width_reasons(d))


def certify_zono3(gm1, gm2, center, eps):
    """Differential verification via diff_net (tightest)."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    merged = diff_net(gm1, gm2)
    op = zono3.interpret(merged)
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
    out = op(x)
    if isinstance(out, DiffExpr3):
        d = out.diff
    else:
        d = out.x - out.y
    return (*d.ublb(), _bound_width_reasons(d))


def certify_zono3_gradlin(gm1, gm2, center, eps):
    """Differential verification via diff_net with gradient linearization."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    merged = diff_net(gm1, gm2)
    op = zono3.interpret_gradlin(merged)
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
    out = op(x)
    if isinstance(out, DiffExpr3):
        d = out.diff
    else:
        d = out.x - out.y
    return (*d.ublb(), _bound_width_reasons(d))


def certify_zonohex(gm1, gm2, center, eps):
    """Differential verification via zonohex."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    merged = diff_net(gm1, gm2)
    op = zonohex.interpret(merged, verbose=True)
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
    out = op(x)
    print("result:", repr(out))
    if isinstance(out, DiffExpr3):
        d = out.diff
    else:
        d = out.x - out.y
    return (*d.ublb(), _bound_width_reasons(d))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_samples(n, data_dir, seed):
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(
            data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(ds), generator=g)[:n].tolist()
        return [(ds[i][0], int(ds[i][1])) for i in indices]
    except Exception:
        g = torch.Generator().manual_seed(seed)
        return [(torch.rand(1, 28, 28, generator=g), -1) for _ in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Differential verification (zonotope): full vs pruned ViT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", default="mnist_transformer.pt")
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=5, dest="n_samples")
    ap.add_argument("--mc-samples", type=int, default=1000, dest="mc_samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=os.path.join(os.getcwd(), "./mnist_data"), dest="data_dir")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false", default=True)
    ap.add_argument("--mean", type=float, default=0.1307)
    ap.add_argument("--std", type=float, default=0.3081)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    vit = build_mnist_vit(args.checkpoint)

    patchify = PatchifyStage(vit, args.normalize, args.mean, args.std).eval()
    gm_patch = onnx_export(patchify, ([1, 28, 28],))
    op_patch = zono.interpret(gm_patch)

    scoring = ScoringModel(vit).eval()
    gm_score = onnx_export(scoring, ([17, 64],))
    op_score = zono.interpret(gm_score)

    samples = load_test_samples(args.n_samples, args.data_dir, args.seed)

    methods = [
        ("zonohex", certify_zonohex),
        ("zono/interval", certify_int_sub),
        ("zono", certify_zono_sub),
        ("zono3", certify_zono3),
        ("zono3/gradlin", certify_zono3_gradlin),
    ]

    print("=" * 75)
    print(f"  Differential Verification (Zono): full vs top-{args.K} pruned ViT")
    print("=" * 75)
    print(f"  eps={args.eps}, K={args.K}, MC={args.mc_samples}")
    print()

    all_results = {name: [] for name, _ in methods}
    all_mc = []
    onnx_cache: dict[bytes, object] = {}
    mask_full = torch.ones(17, 64)
    gm_full = _export_masked_onnx(vit, mask_full, onnx_cache)
    model_full_mc = MaskedModel(vit, mask_full).eval()

    for i, (img, label) in enumerate(samples):
        with torch.no_grad():
            if args.normalize:
                x = (img - args.mean) / args.std
            else:
                x = img
            x = vit.to_patch_embedding(x)
            center = torch.cat((vit.cls_token[0], x), dim=0) + vit.pos_embedding[0]

        full_zono = build_zonotope_no_cat(vit, img, args.eps, op_patch)
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        score_zono = op_score(full_zono)
        ub_sc, lb_sc = score_zono.ublb()

        definite_keep, definite_prune, uncertain = classify_topk(ub_sc, lb_sc, args.K)

        K_remaining = args.K - len(definite_keep)
        if K_remaining < 0:
            K_remaining = 0
            uncertain = set()
        if K_remaining > len(uncertain):
            K_remaining = len(uncertain)

        uncertain_list = sorted(uncertain)
        if len(uncertain_list) == 0 or K_remaining == len(uncertain_list):
            cases = [definite_keep | uncertain]
        elif K_remaining == 0:
            cases = [definite_keep.copy()]
        else:
            from itertools import combinations
            cases = [definite_keep | set(c)
                     for c in combinations(uncertain_list, K_remaining)]

        n_cases = len(cases)

        mc_max = 0.0
        with torch.no_grad():
            for t in range(args.mc_samples):
                torch.manual_seed(t)
                delta = (2 * torch.rand_like(center) - 1) * args.eps
                xp = center + delta
                sc = scoring(xp)
                _, topk = sc.topk(args.K)
                kept_mc = set(topk.tolist())
                mp = torch.zeros(17, 64); mp[0] = 1.0
                for p in kept_mc: mp[p + 1] = 1.0
                model_pruned_mc = MaskedModel(vit, mp).eval()
                diff = model_full_mc(xp) - model_pruned_mc(xp)
                mc_max = max(mc_max, diff.abs().max().item())
        all_mc.append(mc_max)

        print(f"  [{i+1}/{len(samples)}] label={label}, "
              f"keep={len(definite_keep)} prune={len(definite_prune)} "
              f"unc={len(uncertain)} cases={n_cases}")

        for name, fn in methods:
            t0 = time.perf_counter()
            best_d_ub = None
            best_d_lb = None
            best_breakdown = None
            for case_kept in cases:
                mask_pruned = torch.zeros(17, 64)
                mask_pruned[0] = 1.0
                for p in case_kept:
                    mask_pruned[p + 1] = 1.0
                gm_pruned = _export_masked_onnx(vit, mask_pruned, onnx_cache)
                d_ub, d_lb, breakdown = fn(gm_full, gm_pruned, center, args.eps)
                if best_d_ub is None:
                    best_d_ub = d_ub.clone()
                    best_d_lb = d_lb.clone()
                else:
                    best_d_ub = torch.maximum(best_d_ub, d_ub)
                    best_d_lb = torch.minimum(best_d_lb, d_lb)
                best_breakdown = _merge_reason_breakdowns(best_breakdown, breakdown)
            elapsed = time.perf_counter() - t0
            bound = max(best_d_ub.abs().max().item(), best_d_lb.abs().max().item())
            width = (best_d_ub - best_d_lb).mean().item()
            all_results[name].append((bound, elapsed, width))
            print(f"    {name:<15} bound={bound:.4f}  width={width:.4f}  time={elapsed:.1f}s")
            print(f"    {'reasons':<15} {_format_reason_breakdown(best_breakdown or {}, width)}")

        print(f"    {'MC':<15} max|diff|={mc_max:.6f}")

    print()
    print("=" * 75)
    print(f"  SUMMARY: {len(samples)} samples, eps={args.eps}, K={args.K}")
    print("=" * 75)
    avg_mc = sum(all_mc) / len(all_mc)
    print(f"  {'Method':<15} {'Avg Bound':>10} {'Avg Width':>10} {'Avg Time':>9} {'vs Int-Sub':>10}")
    print(f"  {'-'*58}")
    print(f"  {'MC truth':<15} {avg_mc:>10.4f}")

    int_width = None
    for name, _ in methods:
        r = all_results.get(name, [])
        if r:
            avg_b = sum(b for b, _, _ in r) / len(r)
            avg_w = sum(w for _, _, w in r) / len(r)
            avg_t = sum(t for _, t, _ in r) / len(r)
            if int_width is None:
                int_width = avg_w
                vs = ""
            else:
                vs = f"{(1 - avg_w / int_width) * 100:>+.1f}%"
            print(f"  {name:<15} {avg_b:>10.4f} {avg_w:>10.4f} {avg_t:>8.1f}s {vs:>10}")
        else:
            print(f"  {name:<15} {'(no data)':>10}")


if __name__ == "__main__":
    main()
