"""Sweep over K values for the MNIST ViT token-pruning benchmark.

Runs mnist_vit_diff_benchmark for K = 3, 6, 9, 12, 15 with 100 samples each.
Collects all results and prints a cross-K summary at the end.

Usage:
    caffeinate python sweep_k.py
    caffeinate python sweep_k.py --n-samples 50          # lighter
    caffeinate python sweep_k.py --no-binsearch          # sweep only
    caffeinate python sweep_k.py --k-values 4 8 12       # custom K
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

from mnist_vit import build_mnist_vit
from BoundLab.examples.mnist_vit.old_ver.certify import PatchifyStage
from BoundLab.examples.mnist_vit.old_ver.certify_pruned import ScoringModel
from BoundLab.examples.mnist_vit.old_ver.certify_pruned_diff_v2 import load_test_samples

from BoundLab.examples.mnist_vit.old_ver.mnist_vit_diff_benchmark import (
    eps_sweep,
    certifiable_eps_search,
    print_sweep_report,
    print_binsearch_report,
    _stats,
)


def main():
    ap = argparse.ArgumentParser(
        description="Sweep K values for MNIST ViT token-pruning benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", default="mnist_transformer.pt")
    ap.add_argument("--k-values", type=int, nargs="+", default=[3, 6, 9, 12, 15],
                    dest="k_values")
    ap.add_argument("--n-samples", type=int, default=100, dest="n_samples")
    ap.add_argument("--eps-list", type=float, nargs="+",
                    default=[0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064],
                    dest="eps_list")
    ap.add_argument("--d-threshold", type=float, default=1.0, dest="d_threshold")
    ap.add_argument("--search-hi", type=float, default=0.08, dest="search_hi")
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
    print("MNIST ViT Token-Pruning: K-Value Sweep")
    print("=" * 90)
    print(f"  K values:    {args.k_values}")
    print(f"  Samples:     {args.n_samples}")
    print(f"  Eps list:    {args.eps_list}")
    print(f"  D threshold: {args.d_threshold}")
    print()

    torch.manual_seed(args.seed)
    vit = build_mnist_vit(args.checkpoint)

    # Build interpreters once (shared across all K)
    print("Building interpreters...")
    t0 = time.perf_counter()
    patchify = PatchifyStage(vit, args.normalize, args.mean, args.std).eval()
    gm_patch = onnx_export(patchify, ([1, 28, 28],))
    op_patch = zono.interpret(gm_patch)

    scoring = ScoringModel(vit).eval()
    gm_score = onnx_export(scoring, ([17, 64],))
    op_score = zono.interpret(gm_score)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Load samples once (shared across all K)
    samples = load_test_samples(args.n_samples, args.data_dir, args.seed)

    all_sweep_results = {}
    all_binsearch_results = {}
    k_times = {}
    grand_t0 = time.perf_counter()

    for K in args.k_values:
        print("\n")
        print("#" * 90)
        print(f"#  K = {K}  (keeping {K}/16 patches, pruning {16-K})")
        print("#" * 90)

        k_t0 = time.perf_counter()

        if not args.no_sweep:
            print(f"\n--- Eps Sweep (K={K}) ---")
            sweep = eps_sweep(vit, samples, args.eps_list, K, op_patch, op_score)
            print_sweep_report(sweep, K)
            all_sweep_results[K] = sweep

        if not args.no_binsearch:
            print(f"\n--- Binary Search (K={K}, D <= {args.d_threshold}) ---")
            binsearch = certifiable_eps_search(
                vit, samples, K, op_patch, op_score,
                args.d_threshold, args.search_hi,
            )
            print_binsearch_report(binsearch, args.d_threshold, K)
            all_binsearch_results[K] = binsearch

        k_times[K] = time.perf_counter() - k_t0
        print(f"\n  K={K} completed in {k_times[K]:.0f}s")

    # =================================================================
    # Cross-K summary
    # =================================================================
    total_time = time.perf_counter() - grand_t0

    print("\n\n")
    print("=" * 90)
    print("CROSS-K SUMMARY")
    print("=" * 90)

    if all_sweep_results:
        print(f"\n  Mean D across K and eps (Differential):")
        header = f"  {'K':>4} |" + "".join(f" {e:>8}" for e in args.eps_list)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for K in args.k_values:
            if K not in all_sweep_results:
                continue
            row = f"  {K:>4} |"
            for eps in args.eps_list:
                if eps in all_sweep_results[K]:
                    s = _stats(all_sweep_results[K][eps]["diff"]["D"])
                    row += f" {s['mean']:>8.4f}"
                else:
                    row += f" {'N/A':>8}"
            print(row)

        print(f"\n  Mean D across K and eps (Zonotope Sub):")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for K in args.k_values:
            if K not in all_sweep_results:
                continue
            row = f"  {K:>4} |"
            for eps in args.eps_list:
                if eps in all_sweep_results[K]:
                    s = _stats(all_sweep_results[K][eps]["zono"]["D"])
                    row += f" {s['mean']:>8.4f}"
                else:
                    row += f" {'N/A':>8}"
            print(row)

        print(f"\n  Gain (ratio of mean D: Zono / Diff):")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for K in args.k_values:
            if K not in all_sweep_results:
                continue
            row = f"  {K:>4} |"
            for eps in args.eps_list:
                if eps in all_sweep_results[K]:
                    d = _stats(all_sweep_results[K][eps]["diff"]["D"])
                    z = _stats(all_sweep_results[K][eps]["zono"]["D"])
                    g = z["mean"] / (d["mean"] + 1e-30)
                    row += f" {g:>7.1f}x"
                else:
                    row += f" {'N/A':>8}"
            print(row)

    if all_binsearch_results:
        print(f"\n  Mean certifiable eps (D <= {args.d_threshold}):")
        print(f"  {'K':>4} | {'Diff eps':>12} | {'Zono eps':>12} | {'Ratio':>8}")
        print(f"  " + "-" * 45)
        for K in args.k_values:
            if K not in all_binsearch_results:
                continue
            ds = _stats(all_binsearch_results[K]["diff_eps"])
            zs = _stats(all_binsearch_results[K]["zono_eps"])
            ratio = ds["mean"] / (zs["mean"] + 1e-30)
            print(f"  {K:>4} | {ds['mean']:>12.6f} | {zs['mean']:>12.6f} | {ratio:>7.1f}x")

    print(f"\n  Time per K:")
    for K in args.k_values:
        if K in k_times:
            print(f"    K={K:>2}: {k_times[K]:>6.0f}s ({k_times[K]/60:.1f}m)")
    print(f"\n  Total: {total_time:.0f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()