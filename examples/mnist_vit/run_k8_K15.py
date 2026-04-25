"""K=8,15 benchmark: differential vs zonotope subtraction.

Usage:
    cd examples/mnist_vit
    caffeinate python run_k8_k15.py
"""
from __future__ import annotations
import sys, time, warnings, os
from pathlib import Path
from math import comb

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export
from boundlab.diff.net import diff_net
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.zono3 import interpret as diff_interpret

from mnist_vit import build_mnist_vit
from certify import PatchifyStage
from certify_pruned import ScoringModel, build_zonotope_no_cat, classify_topk
from certify_pruned_diff_v2 import MaskedModel, load_test_samples


class _Quiet:
    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self._devnull, self._devnull
        return self
    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._old_out, self._old_err
        self._devnull.close()


K_LIST = [8, 15]
EPS_LIST = [0.01, 0.03, 0.05]
N = 100
MAX_CASES = 20


def verify_one(method, vit, gm_full, center, eps, kept_set):
    mask_p = torch.zeros(17, 64)
    mask_p[0] = 1.0
    for p in kept_set:
        mask_p[p + 1] = 1.0

    with _Quiet():
        gm_p = onnx_export(MaskedModel(vit, mask_p).eval(), ([17, 64],))

        if method == "diff":
            prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
            merged = diff_net(gm_full, gm_p)
            op = diff_interpret(merged)
            x_expr = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
            out = op(x_expr)
            if isinstance(out, DiffExpr3):
                return out.diff.ublb()
            else:
                return (out.x - out.y).ublb()
        else:
            prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
            op1 = zono.interpret(gm_full)
            op2 = zono.interpret(gm_p)
            x_expr = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
            y1 = op1(x_expr)
            y2 = op2(x_expr)
            return (y1 - y2).ublb()


def compute_D(method, vit, gm_full, center, eps, cases):
    best_ub, best_lb = None, None
    for kept in cases:
        ub, lb = verify_one(method, vit, gm_full, center, eps, kept)
        if best_ub is None:
            best_ub, best_lb = ub.clone(), lb.clone()
        else:
            best_ub = torch.maximum(best_ub, ub)
            best_lb = torch.minimum(best_lb, lb)
    return max(best_ub.abs().max().item(), best_lb.abs().max().item())


# Setup
torch.manual_seed(0)
print("Loading model...", flush=True)
with _Quiet():
    vit = build_mnist_vit("mnist_transformer.pt")
    patchify = PatchifyStage(vit, True, 0.1307, 0.3081).eval()
    op_patch = zono.interpret(onnx_export(patchify, ([1, 28, 28],)))
    scoring = ScoringModel(vit).eval()
    op_score = zono.interpret(onnx_export(scoring, ([17, 64],)))
    mask_full = torch.ones(17, 64)
    gm_full = onnx_export(MaskedModel(vit, mask_full).eval(), ([17, 64],))

samples = load_test_samples(N, "./mnist_data", 0)
print("Ready.\n", flush=True)

grand_t0 = time.perf_counter()

for K in K_LIST:
    for eps in EPS_LIST:
        print(f"K={K}  eps={eps}  samples={N}")
        print(f"{'#':>4}  {'label':>5}  {'keep':>4}  {'unc':>3}  {'cases':>5}  "
              f"{'D_diff':>10}  {'D_zono':>10}  {'gain':>6}  {'time':>7}")
        print("-" * 75)

        all_diff, all_zono = [], []
        t_total = time.perf_counter()

        for i, (img, label) in enumerate(samples):
            t0 = time.perf_counter()

            with torch.no_grad():
                x = (img - 0.1307) / 0.3081
                x = vit.to_patch_embedding(x)
                center = torch.cat((vit.cls_token[0], x), dim=0) + vit.pos_embedding[0]

            with _Quiet():
                full_zono_expr = build_zonotope_no_cat(vit, img, eps, op_patch)
                prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
                score_zono_expr = op_score(full_zono_expr)
                ub_sc, lb_sc = score_zono_expr.ublb()

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
                n_combos = comb(len(uncertain_list), K_remaining)
                if n_combos <= MAX_CASES:
                    from itertools import combinations
                    cases = [definite_keep | set(c)
                             for c in combinations(uncertain_list, K_remaining)]
                else:
                    mid = (ub_sc + lb_sc) / 2.0
                    ranked = sorted(uncertain_list, key=lambda t: mid[t].item(), reverse=True)
                    cases = [definite_keep | set(ranked[:K_remaining])]

            D_diff = compute_D("diff", vit, gm_full, center, eps, cases)
            D_zono = compute_D("zono", vit, gm_full, center, eps, cases)
            elapsed = time.perf_counter() - t0

            all_diff.append(D_diff)
            all_zono.append(D_zono)
            gain = D_zono / (D_diff + 1e-30)

            print(f"{i+1:4d}  {label:5d}  {len(definite_keep):4d}  {len(uncertain):3d}  "
                  f"{len(cases):5d}  {D_diff:10.4f}  {D_zono:10.4f}  {gain:5.1f}x  {elapsed:6.1f}s",
                  flush=True)

        print("-" * 75)
        td = torch.tensor(all_diff)
        tz = torch.tensor(all_zono)
        print(f"         {'':>19}  {'Diff':>10}  {'Zono':>10}  {'Gain':>6}")
        print(f"  Mean   {'':>19}  {td.mean().item():10.4f}  {tz.mean().item():10.4f}  "
              f"{tz.mean().item()/(td.mean().item()+1e-30):5.1f}x")
        print(f"  Median {'':>19}  {td.median().item():10.4f}  {tz.median().item():10.4f}  "
              f"{tz.median().item()/(td.median().item()+1e-30):5.1f}x")
        print(f"  Min    {'':>19}  {td.min().item():10.4f}  {tz.min().item():10.4f}")
        print(f"  Max    {'':>19}  {td.max().item():10.4f}  {tz.max().item():10.4f}")
        print(f"  Total time: {time.perf_counter() - t_total:.0f}s\n")

print(f"Grand total: {time.perf_counter() - grand_t0:.0f}s "
      f"({(time.perf_counter() - grand_t0)/60:.1f}m)")