"""MNIST ViT Token-Pruning: Differential Verification Benchmark

Measures how well differential verification bounds the output difference
between a full ViT and its top-K token-pruned variant under L∞ pixel
perturbation.

Primary metric: D = max_i max(|ub_i|, |lb_i|)
    Bound on ||full_model(x) - pruned_model(x)||_∞ over the input region.
    Lower D = tighter bound = stronger equivalence guarantee.

Two experiments:
  1. EPS SWEEP — For each L∞ radius eps, compute D for every sample.
     Report mean/median/min D, plus gain (D_zono / D_diff).

  2. BINARY SEARCH — For a fixed D threshold, binary-search over eps
     to find the largest eps each sample can certify. Reports mean and
     min certifiable eps across samples.

Builds on certify_pruned_diff_v2.py (MaskedModel, case splitting,
patchify pipeline).

Usage:
    cd examples/mnist_vit
    python mnist_vit_diff_benchmark.py --n-samples 10 --K 8             # quick
    python mnist_vit_diff_benchmark.py --n-samples 100 --K 8            # full
    python mnist_vit_diff_benchmark.py --n-samples 100 --K 4            # more pruning
    python mnist_vit_diff_benchmark.py --n-samples 20 --K 8 --d-threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

from mnist_vit import build_mnist_vit
from BoundLab.examples.mnist_vit.old_ver.certify import PatchifyStage
from BoundLab.examples.mnist_vit.old_ver.certify_pruned import ScoringModel, build_zonotope_no_cat, classify_topk
from BoundLab.examples.mnist_vit.old_ver.certify_pruned_diff_v2 import (
    MaskedModel,
    certify_differential,
    certify_zono_sub,
    certify_int_sub,
    load_test_samples,
)


# =====================================================================
# Core: compute D for one sample at one eps
# =====================================================================

def compute_D_for_sample(
    vit, img: Tensor, eps: float, K: int,
    op_patch, op_score,
    method: str = "differential",
) -> float:
    """Compute D = max|diff bound| for full vs top-K pruned ViT.

    Handles the full pipeline: patchify -> score -> classify tokens ->
    case split -> verify each case -> union bounds.

    method: "differential", "zono_sub", or "int_sub"
    Returns D (float), or inf on failure.
    """
    certify_fn = {
        "differential": certify_differential,
        "zono_sub": certify_zono_sub,
        "int_sub": certify_int_sub,
    }[method]

    # Build embedding center and zonotope
    with torch.no_grad():
        x_patches = vit.to_patch_embedding(img)
        center = torch.cat((vit.cls_token[0], x_patches), dim=0) + vit.pos_embedding[0]

    # Get importance score bounds
    full_zono = build_zonotope_no_cat(vit, img, eps, op_patch)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    score_zono = op_score(full_zono)
    ub_sc, lb_sc = score_zono.ublb()

    # Classify tokens
    definite_keep, definite_prune, uncertain = classify_topk(ub_sc, lb_sc, K)

    K_remaining = K - len(definite_keep)
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

    # Full model graph
    mask_full = torch.ones(17, 64)
    gm_full = onnx_export(MaskedModel(vit, mask_full).eval(), ([17, 64],))

    # Verify each case, union bounds
    best_d_ub = None
    best_d_lb = None

    for case_kept in cases:
        mask_pruned = torch.zeros(17, 64)
        mask_pruned[0] = 1.0
        for p in case_kept:
            mask_pruned[p + 1] = 1.0
        gm_pruned = onnx_export(MaskedModel(vit, mask_pruned).eval(), ([17, 64],))

        try:
            d_ub, d_lb = certify_fn(gm_full, gm_pruned, center, eps)
        except Exception:
            return float("inf")

        if best_d_ub is None:
            best_d_ub = d_ub.clone()
            best_d_lb = d_lb.clone()
        else:
            best_d_ub = torch.maximum(best_d_ub, d_ub)
            best_d_lb = torch.minimum(best_d_lb, d_lb)

    if best_d_ub is None:
        return float("inf")

    return max(best_d_ub.abs().max().item(), best_d_lb.abs().max().item())


# =====================================================================
# Experiment 1: Eps sweep
# =====================================================================

def eps_sweep(
    vit, samples, eps_list, K, op_patch, op_score,
) -> dict:
    """For each eps, compute D for every sample under diff and zono_sub."""
    n = len(samples)
    results = {}

    for eps in eps_list:
        print(f"\n  eps = {eps:.4f}")
        diff_Ds, zono_Ds = [], []
        diff_times, zono_times = [], []

        for i, (img, label) in enumerate(samples):
            t0 = time.perf_counter()
            D_diff = compute_D_for_sample(
                vit, img, eps, K, op_patch, op_score, "differential")
            diff_times.append(time.perf_counter() - t0)
            diff_Ds.append(D_diff)

            t0 = time.perf_counter()
            D_zono = compute_D_for_sample(
                vit, img, eps, K, op_patch, op_score, "zono_sub")
            zono_times.append(time.perf_counter() - t0)
            zono_Ds.append(D_zono)

            if (i + 1) % 25 == 0 or i == n - 1:
                gain = D_zono / (D_diff + 1e-30) if D_diff < float("inf") else 0
                print(f"    [{i+1:3d}/{n}]  diff_D={D_diff:.4f}  "
                      f"zono_D={D_zono:.4f}  gain={gain:.1f}x")

        results[eps] = {
            "diff": {"D": diff_Ds, "time": diff_times},
            "zono": {"D": zono_Ds, "time": zono_times},
        }

    return results


# =====================================================================
# Experiment 2: Binary search for max certifiable eps
# =====================================================================

def binary_search_max_eps(
    vit, img, K, op_patch, op_score,
    d_threshold: float,
    method: str,
    lo: float = 0.0,
    hi: float = 0.05,
    tol: float = 1e-5,
    max_iters: int = 16,
) -> float:
    """Find largest eps where D <= d_threshold."""
    D_lo = compute_D_for_sample(vit, img, lo + tol, K, op_patch, op_score, method)
    if D_lo > d_threshold:
        return 0.0
    D_hi = compute_D_for_sample(vit, img, hi, K, op_patch, op_score, method)
    if D_hi <= d_threshold:
        return hi

    for _ in range(max_iters):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        D_mid = compute_D_for_sample(vit, img, mid, K, op_patch, op_score, method)
        if D_mid <= d_threshold:
            lo = mid
        else:
            hi = mid
    return lo


def certifiable_eps_search(
    vit, samples, K, op_patch, op_score,
    d_threshold: float,
    search_hi: float = 0.05,
) -> dict:
    """Per-sample binary search for max eps where D <= d_threshold."""
    n = len(samples)
    diff_epss, zono_epss = [], []

    for i, (img, label) in enumerate(samples):
        e_diff = binary_search_max_eps(
            vit, img, K, op_patch, op_score,
            d_threshold, "differential",
            hi=search_hi,
        )
        diff_epss.append(e_diff)

        e_zono = binary_search_max_eps(
            vit, img, K, op_patch, op_score,
            d_threshold, "zono_sub",
            hi=search_hi,
        )
        zono_epss.append(e_zono)

        if (i + 1) % 10 == 0 or i == n - 1:
            ratio = e_diff / (e_zono + 1e-30)
            print(f"    [{i+1:3d}/{n}]  diff_eps={e_diff:.6f}  "
                  f"zono_eps={e_zono:.6f}  ratio={ratio:.1f}x")

    return {"diff_eps": diff_epss, "zono_eps": zono_epss}


# =====================================================================
# Reporting
# =====================================================================

def _stats(vals):
    t = torch.tensor([v for v in vals if v < float("inf")])
    if len(t) == 0:
        return {k: float("nan") for k in ["mean", "median", "min", "max", "std"]}
    return {
        "mean": t.mean().item(),
        "median": t.median().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "std": t.std().item() if len(t) > 1 else 0.0,
    }


def print_sweep_report(results, K):
    print("\n" + "=" * 90)
    print(f"EPS SWEEP — Certifiable D (full vs top-{K} pruned)")
    print("=" * 90)

    hdr = (f"{'Eps':>8}  | {'Method':<6} | {'Mean D':>10} | {'Median D':>10} | "
           f"{'Min D':>10} | {'Max D':>10} | {'Avg Time':>9}")
    print(hdr)
    print("-" * len(hdr))

    for eps in sorted(results.keys()):
        data = results[eps]
        for method, label in [("diff", "Diff"), ("zono", "Zono")]:
            s = _stats(data[method]["D"])
            t_avg = sum(data[method]["time"]) / len(data[method]["time"])
            print(f"{eps:8.4f}  | {label:<6} | {s['mean']:10.4f} | "
                  f"{s['median']:10.4f} | {s['min']:10.4f} | "
                  f"{s['max']:10.4f} | {t_avg:8.3f}s")

        diff_t = torch.tensor(data["diff"]["D"])
        zono_t = torch.tensor(data["zono"]["D"])
        valid = (diff_t < float("inf")) & (zono_t < float("inf")) & (diff_t > 0)
        if valid.any():
            # Ratio of aggregates (not mean-of-ratios, which inflates on outliers)
            mean_gain = zono_t[valid].mean() / (diff_t[valid].mean() + 1e-30)
            med_gain = zono_t[valid].median() / (diff_t[valid].median() + 1e-30)
            min_gain = zono_t[valid].min() / (diff_t[valid].min() + 1e-30)
            max_gain = zono_t[valid].max() / (diff_t[valid].max() + 1e-30)
            print(f"{'':>8}  | {'Gain':<6} | {mean_gain.item():9.1f}x | "
                  f"{med_gain.item():9.1f}x | {min_gain.item():9.1f}x | "
                  f"{max_gain.item():9.1f}x | {'':>9}")
        print("-" * len(hdr))

    print("\nSummary (mean over samples):")
    print(f"  {'Eps':>8}  {'Diff D':>10}  {'Zono D':>10}  {'Gain':>8}")
    for eps in sorted(results.keys()):
        d = _stats(results[eps]["diff"]["D"])
        z = _stats(results[eps]["zono"]["D"])
        g = z["mean"] / (d["mean"] + 1e-30)
        print(f"  {eps:8.4f}  {d['mean']:10.4f}  {z['mean']:10.4f}  {g:7.1f}x")


def print_binsearch_report(results, d_threshold, K):
    print("\n" + "=" * 90)
    print(f"MAX CERTIFIABLE EPS  (D threshold = {d_threshold}, K = {K})")
    print("=" * 90)

    for method, key in [("Differential", "diff_eps"), ("Zonotope Sub", "zono_eps")]:
        s = _stats(results[key])
        print(f"\n  {method}:")
        print(f"    Mean certifiable eps:   {s['mean']:.6f}")
        print(f"    Median certifiable eps: {s['median']:.6f}")
        print(f"    Min certifiable eps:    {s['min']:.6f}")
        print(f"    Max certifiable eps:    {s['max']:.6f}")

    diff_e = torch.tensor(results["diff_eps"])
    zono_e = torch.tensor(results["zono_eps"])

    print(f"\n  Certification counts at eps thresholds:")
    for e_thresh in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
        n_diff = (diff_e >= e_thresh).sum().item()
        n_zono = (zono_e >= e_thresh).sum().item()
        n_total = len(diff_e)
        print(f"    eps >= {e_thresh:.4f}:  diff={n_diff:3d}/{n_total}  "
              f"zono={n_zono:3d}/{n_total}")

    valid = (diff_e > 0) & (zono_e > 0)
    if valid.any():
        ratios = diff_e[valid] / zono_e[valid]
        print(f"\n  Eps ratio (diff / zono) on "
              f"{valid.sum().item()} jointly-certifiable samples:")
        print(f"    Mean:   {ratios.mean().item():.2f}x")
        print(f"    Median: {ratios.median().item():.2f}x")


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="MNIST ViT token-pruning differential verification benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", default="mnist_transformer.pt")
    ap.add_argument("--K", type=int, default=8,
                    help="Tokens to keep (out of 16 patches)")
    ap.add_argument("--n-samples", type=int, default=100, dest="n_samples")
    ap.add_argument("--eps-list", type=float, nargs="+",
                    default=[0.001, 0.002, 0.004, 0.008, 0.015],
                    dest="eps_list")
    ap.add_argument("--d-threshold", type=float, default=1.0, dest="d_threshold",
                    help="D threshold for binary search")
    ap.add_argument("--search-hi", type=float, default=0.05, dest="search_hi")
    ap.add_argument("--no-sweep", action="store_true", dest="no_sweep")
    ap.add_argument("--no-binsearch", action="store_true", dest="no_binsearch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="./mnist_data", dest="data_dir")
    ap.add_argument("--no-normalize", dest="normalize",
                    action="store_false", default=True)
    ap.add_argument("--mean", type=float, default=0.1307)
    ap.add_argument("--std", type=float, default=0.3081)
    args = ap.parse_args()

    print("=" * 90)
    print(f"MNIST ViT Token-Pruning Differential Verification Benchmark")
    print("=" * 90)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  K (keep):    {args.K} / 16 patches")
    print(f"  Samples:     {args.n_samples}")
    print(f"  Eps list:    {args.eps_list}")
    print(f"  D threshold: {args.d_threshold}")

    torch.manual_seed(args.seed)
    vit = build_mnist_vit(args.checkpoint)

    # Build interpreters once
    print("\nBuilding interpreters...")
    t0 = time.perf_counter()
    patchify = PatchifyStage(vit, args.normalize, args.mean, args.std).eval()
    gm_patch = onnx_export(patchify, ([1, 28, 28],))
    op_patch = zono.interpret(gm_patch)

    scoring = ScoringModel(vit).eval()
    gm_score = onnx_export(scoring, ([17, 64],))
    op_score = zono.interpret(gm_score)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    samples = load_test_samples(args.n_samples, args.data_dir, args.seed)

    total_t0 = time.perf_counter()

    if not args.no_sweep:
        print("\n" + "-" * 90)
        print("Running Experiment 1: Eps Sweep")
        print("-" * 90)
        sweep = eps_sweep(vit, samples, args.eps_list, args.K, op_patch, op_score)
        print_sweep_report(sweep, args.K)

    if not args.no_binsearch:
        print("\n" + "-" * 90)
        print(f"Running Experiment 2: Binary Search (D <= {args.d_threshold})")
        print("-" * 90)
        binsearch = certifiable_eps_search(
            vit, samples, args.K, op_patch, op_score,
            args.d_threshold, args.search_hi,
        )
        print_binsearch_report(binsearch, args.d_threshold, args.K)

    print(f"\nTotal wall-clock time: {time.perf_counter() - total_t0:.1f}s")


if __name__ == "__main__":
    main()