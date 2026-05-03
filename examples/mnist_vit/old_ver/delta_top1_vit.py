"""δ-Top-1 equivalence verification: ViT on MNIST under token pruning.

Loads the MNIST ViT checkpoint, performs top-K token pruning with case
splitting over uncertain tokens, and certifies δ-Top-1 equivalence.

Reports VeryDiff-style tables: certified count per δ level per method,
plus case-splitting statistics.

Usage:
    python examples/mnist_vit/delta_top1_vit.py --eps 0.002 --K 14
    python examples/mnist_vit/delta_top1_vit.py --eps 0.004 --K 8 --n-samples 50
    python examples/mnist_vit/delta_top1_vit.py --eps 0.025 --K 14 --n-samples 100
"""

import argparse
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export
from boundlab.diff.delta_top1 import (
    _collect_epsilons, extract_affine, _solve_top1_lp,
)

from mnist_vit import build_mnist_vit
from BoundLab.examples.mnist_vit.old_ver.certify import PatchifyStage
from BoundLab.examples.mnist_vit.old_ver.certify_pruned import ScoringModel, build_zonotope_no_cat, classify_topk
from BoundLab.examples.mnist_vit.old_ver.certify_pruned_diff_v2 import MaskedModel

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# δ-Top-1 LP from two Exprs
# ---------------------------------------------------------------------------

def delta_top1_from_exprs(expr_x, expr_y, delta):
    """Run the δ-Top-1 LP on two output Exprs. Returns (certified, worst)."""
    t = math.log(delta / (1.0 - delta))
    O = expr_x.shape[0]

    eps_map = _collect_epsilons(expr_x, expr_y)
    c_x, G_x = extract_affine(expr_x, eps_map)
    c_y, G_y = extract_affine(expr_y, eps_map)

    worst = -float('inf')
    for k in range(O):
        for j in range(O):
            if k == j:
                continue
            val = _solve_top1_lp(c_x, G_x, c_y, G_y, k, j, t, O)
            worst = max(worst, val)
            if val > 0:
                return False, worst
    return worst <= 0, worst


# ---------------------------------------------------------------------------
# Three propagation methods returning output expressions
# ---------------------------------------------------------------------------

def propagate_int_sub(gm1, gm2, center, eps):
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    shape = list(center.shape)
    x1 = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out1 = zono.interpret(gm1)(x1)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    x2 = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out2 = zono.interpret(gm2)(x2)
    return out1, out2


def propagate_zono_sub(gm1, gm2, center, eps):
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    shape = list(center.shape)
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out1 = zono.interpret(gm1)(x)
    out2 = zono.interpret(gm2)(x)
    return out1, out2


def propagate_differential(gm1, gm2, center, eps):
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    merged = diff_net(gm1, gm2)
    op = diff_interpret(merged)
    shape = list(center.shape)
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out = op(x)
    if isinstance(out, DiffExpr3):
        return out.x, out.y
    else:
        return out, out


# ---------------------------------------------------------------------------
# Case-split δ-Top-1 verification
# ---------------------------------------------------------------------------

def verify_sample_delta_top1(
    vit, center, eps, K, op_patch, op_score, img,
    prop_fn, deltas, normalize, mean, std,
):
    """Verify one sample across all cases and δ levels.

    Returns:
        certified: dict[float, bool] — per-delta certification result.
        n_cases: int — number of enumerated cases.
        case_stats: dict with definite_keep, definite_prune, uncertain counts.
        dbar: float — max |diff| bound (from ublb).
    """
    # Step 1: Score bounds and token classification
    full_zono = build_zonotope_no_cat(vit, img, eps, op_patch)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    score_zono = op_score(full_zono)
    ub_sc, lb_sc = score_zono.ublb()

    definite_keep, definite_prune, uncertain = classify_topk(ub_sc, lb_sc, K)

    K_remaining = K - len(definite_keep)
    if K_remaining < 0:
        K_remaining = 0; uncertain = set()
    if K_remaining > len(uncertain):
        K_remaining = len(uncertain)

    uncertain_list = sorted(uncertain)
    if len(uncertain_list) == 0 or K_remaining == len(uncertain_list):
        cases = [definite_keep | uncertain]
    elif K_remaining == 0:
        cases = [definite_keep.copy()]
    else:
        cases = [definite_keep | set(c)
                 for c in combinations(uncertain_list, K_remaining)]

    n_cases = len(cases)
    case_stats = {
        'keep': len(definite_keep),
        'prune': len(definite_prune),
        'uncertain': len(uncertain),
    }

    # Step 2: Full model graph
    mask_full = torch.ones(17, 64)
    gm_full = onnx_export(MaskedModel(vit, mask_full).eval(), ([17, 64],))

    # Step 3: For each case, propagate and collect (out_x, out_y)
    # The union across cases: certified only if ALL cases certify.
    # For dbar: take max across cases.
    certified = {d: True for d in deltas}
    worst_dbar = 0.0

    for case_kept in cases:
        mask_pruned = torch.zeros(17, 64)
        mask_pruned[0] = 1.0
        for p in case_kept:
            mask_pruned[p + 1] = 1.0
        gm_pruned = onnx_export(
            MaskedModel(vit, mask_pruned).eval(), ([17, 64],))

        out_x, out_y = prop_fn(gm_full, gm_pruned, center, eps)

        # dbar for this case
        d = out_x - out_y
        d_ub, d_lb = d.ublb()
        dbar_case = max(d_ub.abs().max().item(), d_lb.abs().max().item())
        worst_dbar = max(worst_dbar, dbar_case)

        # δ-Top-1 LP for each delta
        for delta in deltas:
            if not certified[delta]:
                continue  # already failed for a prior case
            cert, _ = delta_top1_from_exprs(out_x, out_y, delta)
            if not cert:
                certified[delta] = False

    return certified, n_cases, case_stats, worst_dbar


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_samples(n, data_dir, seed):
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(
            data_dir, train=False, download=True,
            transform=transforms.ToTensor(),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="mnist_transformer.pt")
    ap.add_argument("--eps", type=float, default=0.025)
    ap.add_argument("--K", type=int, default=14)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="./mnist_data")
    ap.add_argument("--mean", type=float, default=0.1307)
    ap.add_argument("--std", type=float, default=0.3081)
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.6, 0.8, 0.9, 0.95, 0.99, 0.999])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    vit = build_mnist_vit(args.checkpoint)

    patchify = PatchifyStage(vit, True, args.mean, args.std).eval()
    gm_patch = onnx_export(patchify, ([1, 28, 28],))
    op_patch = zono.interpret(gm_patch)

    scoring = ScoringModel(vit).eval()
    gm_score = onnx_export(scoring, ([17, 64],))
    op_score = zono.interpret(gm_score)

    samples = load_test_samples(args.n_samples, args.data_dir, args.seed)
    N = len(samples)

    methods = [
        ("Int-Sub", propagate_int_sub),
        ("Zono-Sub", propagate_zono_sub),
        ("Differential", propagate_differential),
    ]

    print(f"{'='*75}")
    print(f"  δ-Top-1 Verification: full vs top-{args.K} pruned ViT")
    print(f"{'='*75}")
    print(f"  eps={args.eps}, K={args.K}, N={N}")
    print(f"  deltas={args.deltas}")
    print()

    # Results: method -> delta -> count
    results = {name: {d: 0 for d in args.deltas} for name, _ in methods}
    all_dbar = {name: [] for name, _ in methods}
    all_cases = []
    all_stats = []
    times = {name: 0.0 for name, _ in methods}

    for i, (img, label) in enumerate(samples):
        # Compute embedding center
        with torch.no_grad():
            x = (img - args.mean) / args.std
            x = vit.to_patch_embedding(x)
            center = torch.cat((vit.cls_token[0], x), dim=0) + \
                     vit.pos_embedding[0]

        first_method = True
        for name, prop_fn in methods:
            t0 = time.perf_counter()
            try:
                certified, n_cases, stats, dbar = verify_sample_delta_top1(
                    vit, center, args.eps, args.K,
                    op_patch, op_score, img,
                    prop_fn, args.deltas,
                    True, args.mean, args.std,
                )

                for delta in args.deltas:
                    if certified[delta]:
                        results[name][delta] += 1

                all_dbar[name].append(dbar)

                if first_method:
                    all_cases.append(n_cases)
                    all_stats.append(stats)
                    first_method = False

            except Exception as e:
                all_dbar[name].append(float('inf'))
                import traceback; traceback.print_exc()

            elapsed = time.perf_counter() - t0
            times[name] += elapsed

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{N}] label={label} "
                  f"cases={all_cases[-1] if all_cases else '?'} "
                  f"keep={all_stats[-1]['keep'] if all_stats else '?'} "
                  f"unc={all_stats[-1]['uncertain'] if all_stats else '?'}")

    # =====================================================================
    # Table 1: δ-Top-1 certified count (VeryDiff Table 2 style)
    # =====================================================================

    print(f"\n{'='*75}")
    print(f"  TABLE 1: δ-Top-1 Certified Count (N={N})")
    print(f"{'='*75}")
    print(f"  {'δ':>8} {'t':>8}", end="")
    for name, _ in methods:
        print(f" {name:>14}", end="")
    print()
    print(f"  {'-'*8} {'-'*8}", end="")
    for _ in methods:
        print(f" {'-'*14}", end="")
    print()

    for delta in args.deltas:
        t = math.log(delta / (1.0 - delta))
        print(f"  {delta:>8.4f} {t:>8.3f}", end="")
        for name, _ in methods:
            c = results[name][delta]
            print(f" {c:>8d}/{N:<4d}", end="")
        print()

    # =====================================================================
    # Table 2: Case splitting statistics
    # =====================================================================

    print(f"\n{'='*75}")
    print(f"  TABLE 2: Token Classification Statistics")
    print(f"{'='*75}")
    if all_stats:
        avg_keep = sum(s['keep'] for s in all_stats) / len(all_stats)
        avg_prune = sum(s['prune'] for s in all_stats) / len(all_stats)
        avg_unc = sum(s['uncertain'] for s in all_stats) / len(all_stats)
        avg_cases = sum(all_cases) / len(all_cases)
        print(f"  Avg definite-keep:  {avg_keep:.1f}")
        print(f"  Avg definite-prune: {avg_prune:.1f}")
        print(f"  Avg uncertain:      {avg_unc:.1f}")
        print(f"  Avg cases:          {avg_cases:.1f}")

    # =====================================================================
    # Table 3: Bound tightness and timing
    # =====================================================================

    print(f"\n{'='*75}")
    print(f"  TABLE 3: Bound Tightness and Timing")
    print(f"{'='*75}")
    print(f"  {'Method':<15} {'Avg d_bar':>10} {'Total Time':>12}")
    print(f"  {'-'*40}")
    for name, _ in methods:
        db = all_dbar[name]
        finite = [x for x in db if x < float('inf')]
        avg = sum(finite) / len(finite) if finite else float('inf')
        print(f"  {name:<15} {avg:>10.4f} {times[name]:>11.1f}s")

    # =====================================================================
    # Table 4: Min certifiable δ (margin-based estimate)
    # =====================================================================

    print(f"\n{'='*75}")
    print(f"  TABLE 4: Min Certifiable δ (margin-based estimate)")
    print(f"{'='*75}")
    print(f"  {'Method':<15} {'Mean δ_min':>12} {'Worst δ_min':>12}")
    print(f"  {'-'*42}")
    for name, _ in methods:
        db = all_dbar[name]
        finite = [x for x in db if x < float('inf')]
        if finite:
            d_mins = [1.0 / (1.0 + math.exp(-2 * d)) for d in finite]
            print(f"  {name:<15} {sum(d_mins)/len(d_mins):>12.6f} "
                  f"{max(d_mins):>12.6f}")


if __name__ == "__main__":
    main()