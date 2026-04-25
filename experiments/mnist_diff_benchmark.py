"""MNIST Differential Verification Benchmark

Compares differential zonotope verification vs zonotope subtraction baseline
on MNIST pruned-network equivalence.

Primary metric: D = max_i max(|ub_i|, |lb_i|)
    The smallest ε for which we can certify ε-equivalence over the input region.
    Lower D = stronger guarantee.

Two experiments:
  1. RADIUS SWEEP — For each L∞ input radius ν, compute D for every sample.
     Report mean/median/min D per radius, plus the differential gain (D_zono / D_diff).

  2. BINARY SEARCH — For a fixed ε threshold, binary-search over input radius
     to find the largest ν at which each sample can be certified (D ≤ ε).
     Report mean and min certifiable radius across samples.

Requirements:
    pip install torchvision boundlab onnx-ir beartype onnxscript

Usage:
    python experiments/mnist_diff_benchmark.py                    # full run
    python experiments/mnist_diff_benchmark.py --n-samples 10     # quick test
    python experiments/mnist_diff_benchmark.py --eps-threshold 0.5 # tighter ε
    python experiments/mnist_diff_benchmark.py --no-binsearch     # sweep only
    python experiments/mnist_diff_benchmark.py --no-sweep         # binary search only
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.net import diff_net
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.zono3 import interpret as diff_interpret


NETS_DIR = Path(__file__).resolve().parent.parent / "compare" / "verydiff" / "examples" / "nets"
NET1_PATH = NETS_DIR / "mnist_relu_3_100.onnx"
NET2_PATH = NETS_DIR / "mnist_relu_3_100_pruned5.onnx"


# =====================================================================
# Load MNIST samples
# =====================================================================

def load_mnist_samples(n: int, seed: int = 42) -> list[torch.Tensor]:
    """Load n MNIST test images as flattened [0,1] tensors of shape (784,).

    Downloads the dataset on first call via torchvision.
    """
    from torchvision import datasets, transforms

    ds = datasets.MNIST(
        root=str(Path(__file__).resolve().parent.parent / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),  # [0,255] -> [0,1], shape (1,28,28)
    )

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=rng)[:n]
    samples = []
    for idx in indices:
        img, _ = ds[int(idx.item())]
        samples.append(img.flatten())  # (784,)
    return samples


# =====================================================================
# Core verification routines
# =====================================================================

def compute_D_differential(
    merged_graph, center: torch.Tensor, radius: float,
) -> tuple[float, float, torch.Tensor]:
    """Differential verification -> (max_D, mean_D, per_dim_D)."""
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()
    op = diff_interpret(merged_graph)
    x_expr = expr.ConstVal(center) + radius * expr.LpEpsilon([784])
    triple = DiffExpr3(x_expr, x_expr, expr.ConstVal(torch.zeros(784)))
    out = op(triple)
    ub, lb = out.diff.ublb()
    per_dim = torch.maximum(ub.abs(), lb.abs())
    return per_dim.max().item(), per_dim.mean().item(), per_dim


def compute_D_zonosub(
    net1_path: Path, net2_path: Path, center: torch.Tensor, radius: float,
) -> tuple[float, float, torch.Tensor]:
    """Zonotope subtraction baseline -> (max_D, mean_D, per_dim_D)."""
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()
    op1 = zono.interpret(str(net1_path))
    op2 = zono.interpret(str(net2_path))
    x_shared = expr.ConstVal(center) + radius * expr.LpEpsilon([784])
    y1 = op1(x_shared)
    y2 = op2(x_shared)
    ub, lb = (y1 - y2).ublb()
    per_dim = torch.maximum(ub.abs(), lb.abs())
    return per_dim.max().item(), per_dim.mean().item(), per_dim


# =====================================================================
# Monte Carlo ground truth
# =====================================================================

def monte_carlo_D(
    net1_path: Path, net2_path: Path,
    center: torch.Tensor, radius: float,
    n_samples: int = 2000,
) -> tuple[float, float]:
    """Sample-based estimate of true max |f1(x)-f2(x)| over L-inf ball.

    Returns (max_D, mean_D).
    """
    import onnx_ir as ir

    def _load_params(path):
        model = ir.load(str(path))
        params = {}
        for name, val in model.graph.initializers.items():
            params[name] = torch.from_numpy(val.const_value.numpy()).float()
        return params

    def _forward(params, x):
        h = params["W0"] @ x + params["B0"]
        h = torch.relu(h)
        h = params["W1"] @ h + params["B1"]
        h = torch.relu(h)
        h = params["W2"] @ h + params["B2"]
        h = torch.relu(h)
        h = params["W3"] @ h + params["B3"]
        return h

    p1 = _load_params(net1_path)
    p2 = _load_params(net2_path)

    max_D = 0.0
    sum_mean = 0.0
    with torch.no_grad():
        for _ in range(n_samples):
            noise = (torch.rand_like(center) * 2 - 1) * radius
            x = (center + noise).clamp(0.0, 1.0)
            diff = (_forward(p1, x) - _forward(p2, x)).abs()
            max_D = max(max_D, diff.max().item())
            sum_mean += diff.mean().item()
    return max_D, sum_mean / n_samples


# =====================================================================
# Binary search for max certifiable radius
# =====================================================================

def _binsearch(certify_fn, eps_threshold, lo, hi, tol, max_iters):
    """Generic binary search: find max radius where certify_fn(r) <= threshold."""
    D_lo = certify_fn(lo + tol)
    if D_lo > eps_threshold:
        return 0.0
    D_hi = certify_fn(hi)
    if D_hi <= eps_threshold:
        return hi
    for _ in range(max_iters):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        if certify_fn(mid) <= eps_threshold:
            lo = mid
        else:
            hi = mid
    return lo


def binary_search_max_radius(
    merged_graph, center: torch.Tensor, eps_threshold: float,
    lo: float = 0.0, hi: float = 0.5, tol: float = 1e-4, max_iters: int = 20,
) -> float:
    """Max input radius where D_diff <= eps_threshold."""
    def certify(r):
        d, _, _ = compute_D_differential(merged_graph, center, r)
        return d
    return _binsearch(certify, eps_threshold, lo, hi, tol, max_iters)


def binary_search_max_radius_zonosub(
    net1_path: Path, net2_path: Path,
    center: torch.Tensor, eps_threshold: float,
    lo: float = 0.0, hi: float = 0.5, tol: float = 1e-4, max_iters: int = 20,
) -> float:
    """Max input radius where D_zono <= eps_threshold."""
    def certify(r):
        d, _, _ = compute_D_zonosub(net1_path, net2_path, center, r)
        return d
    return _binsearch(certify, eps_threshold, lo, hi, tol, max_iters)


# =====================================================================
# Experiment 1: Radius sweep
# =====================================================================

def radius_sweep(
    samples: list[torch.Tensor],
    radii: list[float],
    merged_graph,
    run_mc: bool = True,
    mc_samples: int = 500,
) -> dict:
    """For each radius, compute D for every sample under both methods."""
    n = len(samples)
    results = {}

    for radius in radii:
        print(f"\n  radius = {radius:.4f}")
        diff_maxDs, diff_avgDs, diff_times = [], [], []
        zono_maxDs, zono_avgDs, zono_times = [], [], []
        mc_maxDs, mc_avgDs = [], []

        for i, center in enumerate(samples):
            t0 = time.perf_counter()
            dmax, davg, _ = compute_D_differential(merged_graph, center, radius)
            diff_times.append(time.perf_counter() - t0)
            diff_maxDs.append(dmax)
            diff_avgDs.append(davg)

            t0 = time.perf_counter()
            zmax, zavg, _ = compute_D_zonosub(NET1_PATH, NET2_PATH, center, radius)
            zono_times.append(time.perf_counter() - t0)
            zono_maxDs.append(zmax)
            zono_avgDs.append(zavg)

            if run_mc:
                mc_max, mc_avg = monte_carlo_D(
                    NET1_PATH, NET2_PATH, center, radius, n_samples=mc_samples,
                )
                mc_maxDs.append(mc_max)
                mc_avgDs.append(mc_avg)

            if (i + 1) % 25 == 0 or i == n - 1:
                print(f"    [{i+1:3d}/{n}]  diff_D={dmax:.4f}  zono_D={zmax:.4f}  "
                      f"gain={zmax / (dmax + 1e-30):.1f}x")

        entry = {
            "diff": {"max_D": diff_maxDs, "avg_D": diff_avgDs, "time": diff_times},
            "zono": {"max_D": zono_maxDs, "avg_D": zono_avgDs, "time": zono_times},
        }
        if run_mc:
            entry["mc"] = {"max_D": mc_maxDs, "avg_D": mc_avgDs}
        results[radius] = entry

    return results


# =====================================================================
# Experiment 2: Binary search for max certifiable radius
# =====================================================================

def certifiable_radius_search(
    samples: list[torch.Tensor],
    merged_graph,
    eps_threshold: float,
    search_hi: float = 0.3,
) -> dict:
    """Per-sample binary search for max radius where D <= eps_threshold."""
    n = len(samples)
    diff_radii, zono_radii = [], []

    for i, center in enumerate(samples):
        r_diff = binary_search_max_radius(
            merged_graph, center, eps_threshold,
            lo=0.0, hi=search_hi, tol=1e-4, max_iters=16,
        )
        diff_radii.append(r_diff)

        r_zono = binary_search_max_radius_zonosub(
            NET1_PATH, NET2_PATH, center, eps_threshold,
            lo=0.0, hi=search_hi, tol=1e-4, max_iters=16,
        )
        zono_radii.append(r_zono)

        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"    [{i+1:3d}/{n}]  diff_r={r_diff:.5f}  zono_r={r_zono:.5f}  "
                  f"ratio={r_diff / (r_zono + 1e-30):.1f}x")

    return {"diff_radii": diff_radii, "zono_radii": zono_radii}


# =====================================================================
# Reporting
# =====================================================================

def _stats(vals: list[float]) -> dict:
    t = torch.tensor(vals)
    return {
        "mean": t.mean().item(),
        "median": t.median().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "std": t.std().item() if len(vals) > 1 else 0.0,
    }


def print_sweep_report(results: dict):
    print("\n" + "=" * 94)
    print("EXPERIMENT 1: RADIUS SWEEP  —  Certifiable eps (distance bound D)")
    print("=" * 94)

    hdr = (f"{'Radius':>8}  | {'Method':<6} | {'Mean D':>10} | {'Median D':>10} | "
           f"{'Min D':>10} | {'Max D':>10} | {'Avg Time':>9}")
    print(hdr)
    print("-" * len(hdr))

    for radius in sorted(results.keys()):
        data = results[radius]
        for method, label in [("diff", "Diff"), ("zono", "Zono")]:
            s = _stats(data[method]["max_D"])
            t_avg = sum(data[method]["time"]) / len(data[method]["time"])
            print(f"{radius:8.4f}  | {label:<6} | {s['mean']:10.4f} | {s['median']:10.4f} | "
                  f"{s['min']:10.4f} | {s['max']:10.4f} | {t_avg:8.3f}s")

        diff_t = torch.tensor(data["diff"]["max_D"])
        zono_t = torch.tensor(data["zono"]["max_D"])
        gains = zono_t / (diff_t + 1e-30)
        print(f"{'':>8}  | {'Gain':<6} | {gains.mean().item():9.1f}x | "
              f"{gains.median().item():9.1f}x | "
              f"{gains.min().item():9.1f}x | {gains.max().item():9.1f}x | {'':>9}")

        if "mc" in data:
            mc_t = torch.tensor(data["mc"]["max_D"])
            btr_d = (mc_t / (diff_t + 1e-30)).mean().item()
            btr_z = (mc_t / (zono_t + 1e-30)).mean().item()
            sound_d = (diff_t >= mc_t - 1e-3).all().item()
            sound_z = (zono_t >= mc_t - 1e-3).all().item()
            print(f"{'':>8}  | {'MC':>6} | {'BTR_d=' + f'{btr_d:.2f}':>10} | "
                  f"{'BTR_z=' + f'{btr_z:.2f}':>10} | "
                  f"{'snd_d=' + str(sound_d):>10} | "
                  f"{'snd_z=' + str(sound_z):>10} | {'':>9}")
        print("-" * len(hdr))

    print("\nSummary (mean over samples):")
    print(f"  {'Radius':>8}  {'Diff D':>10}  {'Zono D':>10}  {'Gain':>8}")
    for radius in sorted(results.keys()):
        d = _stats(results[radius]["diff"]["max_D"])
        z = _stats(results[radius]["zono"]["max_D"])
        print(f"  {radius:8.4f}  {d['mean']:10.4f}  {z['mean']:10.4f}  "
              f"{z['mean'] / (d['mean'] + 1e-30):7.1f}x")


def print_binsearch_report(results: dict, eps_threshold: float):
    print("\n" + "=" * 94)
    print(f"EXPERIMENT 2: MAX CERTIFIABLE RADIUS  (eps threshold = {eps_threshold})")
    print("=" * 94)

    for method, key in [("Differential", "diff_radii"), ("Zonotope Sub", "zono_radii")]:
        s = _stats(results[key])
        print(f"\n  {method}:")
        print(f"    Mean certifiable radius:   {s['mean']:.6f}")
        print(f"    Median certifiable radius: {s['median']:.6f}")
        print(f"    Min certifiable radius:    {s['min']:.6f}")
        print(f"    Max certifiable radius:    {s['max']:.6f}")
        print(f"    Std:                       {s['std']:.6f}")

    diff_r = torch.tensor(results["diff_radii"])
    zono_r = torch.tensor(results["zono_radii"])

    print(f"\n  Certification counts at radius thresholds:")
    for r_thresh in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        n_diff = (diff_r >= r_thresh).sum().item()
        n_zono = (zono_r >= r_thresh).sum().item()
        n_total = len(diff_r)
        print(f"    r >= {r_thresh:.3f}:  diff={n_diff:3d}/{n_total}  "
              f"zono={n_zono:3d}/{n_total}")

    valid = (diff_r > 0) & (zono_r > 0)
    if valid.any():
        ratios = diff_r[valid] / zono_r[valid]
        print(f"\n  Radius ratio (diff / zono) on "
              f"{valid.sum().item()} jointly-certifiable samples:")
        print(f"    Mean:   {ratios.mean().item():.2f}x")
        print(f"    Median: {ratios.median().item():.2f}x")
        print(f"    Min:    {ratios.min().item():.2f}x")
        print(f"    Max:    {ratios.max().item():.2f}x")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MNIST differential verification benchmark",
    )
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of MNIST test images (default: 100)")
    parser.add_argument("--eps-threshold", type=float, default=1.0,
                        help="eps threshold for binary search (default: 1.0)")
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                        help="L-inf input radii for the sweep")
    parser.add_argument("--search-hi", type=float, default=0.15,
                        help="Upper bound for binary search (default: 0.15)")
    parser.add_argument("--mc-samples", type=int, default=500,
                        help="Monte Carlo samples per image (default: 500)")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip Monte Carlo soundness checks")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip radius sweep")
    parser.add_argument("--no-binsearch", action="store_true",
                        help="Skip binary search")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not NET1_PATH.exists() or not NET2_PATH.exists():
        print(f"ERROR: MNIST models not found at {NETS_DIR}")
        print(f"  Expected: mnist_relu_3_100.onnx and mnist_relu_3_100_pruned5.onnx")
        return

    print("=" * 94)
    print("MNIST Differential Verification Benchmark")
    print("=" * 94)
    print(f"  Net1:          {NET1_PATH.name}")
    print(f"  Net2:          {NET2_PATH.name}")
    print(f"  Samples:       {args.n_samples}")
    print(f"  Radii:         {args.radii}")
    print(f"  eps threshold: {args.eps_threshold}")
    print(f"  Search hi:     {args.search_hi}")
    print(f"  Monte Carlo:   {not args.no_mc} ({args.mc_samples} samples)")

    # Merge models once
    print("\nMerging ONNX models for differential interpretation...")
    t0 = time.perf_counter()
    merged = diff_net(str(NET1_PATH), str(NET2_PATH))
    print(f"  Done in {time.perf_counter() - t0:.2f}s")

    # Load MNIST
    print(f"\nLoading {args.n_samples} MNIST test images (seed={args.seed})...")
    samples = load_mnist_samples(args.n_samples, seed=args.seed)
    sparsity = (samples[0] == 0).float().mean()
    print(f"  Loaded.  Sample 0 sparsity: {sparsity:.0%} zeros")

    total_t0 = time.perf_counter()

    # Experiment 1
    if not args.no_sweep:
        print("\n" + "-" * 94)
        print("Running Experiment 1: Radius Sweep")
        print("-" * 94)
        sweep = radius_sweep(
            samples, args.radii, merged,
            run_mc=not args.no_mc, mc_samples=args.mc_samples,
        )
        print_sweep_report(sweep)

    # Experiment 2
    if not args.no_binsearch:
        print("\n" + "-" * 94)
        print(f"Running Experiment 2: Binary Search (eps <= {args.eps_threshold})")
        print("-" * 94)
        binsearch = certifiable_radius_search(
            samples, merged, args.eps_threshold, search_hi=args.search_hi,
        )
        print_binsearch_report(binsearch, args.eps_threshold)

    print(f"\nTotal wall-clock time: {time.perf_counter() - total_t0:.1f}s")


if __name__ == "__main__":
    main()