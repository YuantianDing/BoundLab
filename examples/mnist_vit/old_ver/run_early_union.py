"""Experiment: early union with end-to-end differential verification.

Two models f1 (full, mask=ones) and f2 (pruned).  They share the same input
and diverge at the pruning mask.  f2 splits into multiple cases (one per
possible kept-set).  For each case, differential verification tracks f1 vs
f2_case through attention.  After attention, all f2 cases merge back into a
single f2 via zonotope hull, then differential verification continues through
the tail (FF -> pool -> head) with f1 untouched.

Compares:

  1. **Diff baseline**: Per-case diff_net on full MaskedModel, union at end.
     N full differential propagations.

  2. **Zono-sub baseline**: Per-case shared zonotope, subtract, union at end.
     N full zonotope propagations.

  3. **Early union + diff**: Differential through attention (base case),
     differential deviations for other cases, hull merge, differential
     through tail.  N attention-only diff + 1 tail diff propagation.

Usage:
    cd examples/mnist_vit
    python run_early_union.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from itertools import combinations
from math import comb
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
from torch import nn, Tensor

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
from BoundLab.examples.mnist_vit.old_ver.certify import PatchifyStage
from BoundLab.examples.mnist_vit.old_ver.certify_pruned import ScoringModel, build_zonotope_no_cat, classify_topk
from BoundLab.examples.mnist_vit.old_ver.certify_pruned_diff_v2 import MaskedModel, load_test_samples


# ---------------------------------------------------------------------------
# Split sub-models
# ---------------------------------------------------------------------------

class MaskedAttentionStage(nn.Module):
    """(17, 64) -> (17, 64): mask * input -> attention + residual."""

    def __init__(self, vit, mask: Tensor):
        super().__init__()
        attn_block = vit.transformer.layers[0][0]
        self.attn_norm = attn_block.fn.norm
        self.attn = attn_block.fn.fn
        self.heads = self.attn.heads
        self.dim_head = self.attn.dim_head
        self.scale = self.attn.scale
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
        return residual + out


class PostAttentionStage(nn.Module):
    """(17, 64) -> (10,): FF block -> pool -> MLP head."""

    def __init__(self, vit):
        super().__init__()
        self.ff_block = vit.transformer.layers[0][1]
        self.pool = vit.pool
        self.mlp_head = vit.mlp_head

    def forward(self, x: Tensor) -> Tensor:
        x = self.ff_block(x)
        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)


# ---------------------------------------------------------------------------
# Quiet context
# ---------------------------------------------------------------------------

class _Quiet:
    def __enter__(self):
        self._devnull = open(os.devnull, "w")
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self._devnull, self._devnull
        return self

    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._old_out, self._old_err
        self._devnull.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pruned_mask(kept_set):
    mask = torch.zeros(17, 64)
    mask[0] = 1.0
    for p in kept_set:
        mask[p + 1] = 1.0
    return mask


def D_from_ublb(ub, lb):
    return max(ub.abs().max().item(), lb.abs().max().item())


# ---------------------------------------------------------------------------
# Method 1: Differential baseline (per-case, full model)
# ---------------------------------------------------------------------------

def certify_diff(vit, gm_full, center, eps, cases):
    """Per-case diff_net on full MaskedModel.  Union at end."""
    best_ub, best_lb = None, None

    for kept_set in cases:
        with _Quiet():
            gm_p = onnx_export(
                MaskedModel(vit, _build_pruned_mask(kept_set)).eval(),
                ([17, 64],),
            )

        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()
        merged = diff_net(gm_full, gm_p)
        op = diff_interpret(merged)
        x_expr = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
        out = op(x_expr)
        if isinstance(out, DiffExpr3):
            ub, lb = out.diff.ublb()
        else:
            ub, lb = (out.x - out.y).ublb()

        if best_ub is None:
            best_ub, best_lb = ub.clone(), lb.clone()
        else:
            best_ub = torch.maximum(best_ub, ub)
            best_lb = torch.minimum(best_lb, lb)

    return best_ub, best_lb


# ---------------------------------------------------------------------------
# Method 2: Zono-sub baseline (per-case, full model)
# ---------------------------------------------------------------------------

def certify_zono_sub(vit, gm_full, center, eps, cases):
    """Per-case shared zonotope, full - pruned.  Union at end."""
    best_ub, best_lb = None, None

    for kept_set in cases:
        with _Quiet():
            gm_p = onnx_export(
                MaskedModel(vit, _build_pruned_mask(kept_set)).eval(),
                ([17, 64],),
            )

        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()
        op1 = zono.interpret(gm_full)
        op2 = zono.interpret(gm_p)
        x_expr = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
        y1 = op1(x_expr)
        y2 = op2(x_expr)
        ub, lb = (y1 - y2).ublb()

        if best_ub is None:
            best_ub, best_lb = ub.clone(), lb.clone()
        else:
            best_ub = torch.maximum(best_ub, ub)
            best_lb = torch.minimum(best_lb, lb)

    return best_ub, best_lb


# ---------------------------------------------------------------------------
# Method 3: Early union with end-to-end differential
# ---------------------------------------------------------------------------

def certify_early_union_diff(vit, gm_attn_full, gm_tail, center, eps, cases):
    """End-to-end differential with zonotope hull merge after attention.

    1. Shared x_expr (one LpEpsilon).
    2. Pick first case as base.  diff_net(attn_full, attn_base) ->
       DiffExpr3(mid_full, mid_base, diff_base): tight differential
       tracking of f1 vs f2_base through attention.
    3. For each other case: diff_net(attn_base, attn_case_i) for tight
       deviation bounds between f2 variants.
    4. Zonotope hull on f2 side.  diff component adjusted to match.
       Hull includes the base case (zero deviation).
    5. DiffExpr3(mid_full, z_union, diff_hull) -> diff_interpret(tail).
    """
    cases_list = list(cases)
    base_kept = cases_list[0]

    # --- Step 1: shared input ---
    x_expr = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))

    # --- Step 2: differential through attention (f1 vs f2_base) ---
    with _Quiet():
        gm_attn_base = onnx_export(
            MaskedAttentionStage(vit, _build_pruned_mask(base_kept)).eval(),
            ([17, 64],),
        )

    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()
    merged_attn = diff_net(gm_attn_full, gm_attn_base)
    op_attn_diff = diff_interpret(merged_attn)
    attn_out = op_attn_diff(x_expr)

    if isinstance(attn_out, DiffExpr3):
        mid_full = attn_out.x
        mid_base = attn_out.y
        diff_base = attn_out.diff
    else:
        mid_full = attn_out.x
        mid_base = attn_out.y
        diff_base = mid_full - mid_base

    # --- Step 3-4: deviation bounds and hull ---
    if len(cases_list) > 1:
        # Deviation bounds: diff_net(base, case_i) for each other case.
        # Initialize with zeros to include base case in hull.
        dev_ub_all = None
        dev_lb_all = None

        for kept_set in cases_list[1:]:
            with _Quiet():
                gm_attn_i = onnx_export(
                    MaskedAttentionStage(
                        vit, _build_pruned_mask(kept_set)
                    ).eval(),
                    ([17, 64],),
                )

            prop._UB_CACHE.clear()
            prop._LB_CACHE.clear()
            merged_dev = diff_net(gm_attn_base, gm_attn_i)
            op_dev = diff_interpret(merged_dev)
            dev_out = op_dev(x_expr)

            if isinstance(dev_out, DiffExpr3):
                d_ub, d_lb = dev_out.diff.ublb()
            else:
                d_ub, d_lb = (dev_out.x - dev_out.y).ublb()

            if dev_ub_all is None:
                # Include base case (zero deviation) via max/min with zeros
                dev_ub_all = torch.maximum(d_ub, torch.zeros_like(d_ub))
                dev_lb_all = torch.minimum(d_lb, torch.zeros_like(d_lb))
            else:
                dev_ub_all = torch.maximum(dev_ub_all, d_ub)
                dev_lb_all = torch.minimum(dev_lb_all, d_lb)

        # Hull of f2 cases:
        #   deviation = base - case_i, so case_i = base - deviation
        #   case_i ranges in [base - dev_ub, base - dev_lb]
        #   z_union = base - corr_center ± corr_hw
        corr_center = (dev_ub_all + dev_lb_all) / 2.0
        corr_hw = (dev_ub_all - dev_lb_all) / 2.0

        new_eps = expr.LpEpsilon(list(mid_base.shape))
        z_union = mid_base - expr.ConstVal(corr_center) + corr_hw * new_eps

        # diff = f1 - f2 = f1 - (base - corr_center ± corr_hw)
        #      = diff_base + corr_center ∓ corr_hw
        diff_hull = diff_base + expr.ConstVal(corr_center) - corr_hw * new_eps
    else:
        z_union = mid_base
        diff_hull = diff_base

    # --- Step 5: differential through tail ---
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()
    op_tail = diff_interpret(gm_tail)
    tail_input = DiffExpr3(mid_full, z_union, diff_hull)
    out = op_tail(tail_input)

    if isinstance(out, DiffExpr3):
        return out.diff.ublb()
    else:
        return (out.x - out.y).ublb()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

K_LIST = [15]
EPS_LIST = [0.01, 0.03, 0.05]
N = 100
MAX_CASES = 20


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)

    print("=" * 105)
    print("  Early Union Experiment (end-to-end differential)")
    print("  Diff baseline vs Zono-sub baseline vs Early union + diff")
    print("=" * 105)
    print()

    print("Loading model...", flush=True)
    with _Quiet():
        vit = build_mnist_vit("mnist_transformer.pt")
        patchify = PatchifyStage(vit, True, 0.1307, 0.3081).eval()
        op_patch = zono.interpret(onnx_export(patchify, ([1, 28, 28],)))
        scoring = ScoringModel(vit).eval()
        op_score = zono.interpret(onnx_export(scoring, ([17, 64],)))
        mask_full = torch.ones(17, 64)
        gm_full = onnx_export(MaskedModel(vit, mask_full).eval(), ([17, 64],))
        gm_attn_full = onnx_export(
            MaskedAttentionStage(vit, mask_full).eval(), ([17, 64],)
        )
        gm_tail = onnx_export(PostAttentionStage(vit).eval(), ([17, 64],))

    samples = load_test_samples(N, "../mnist_data", 0)
    print(f"Ready. {len(samples)} samples loaded.\n", flush=True)

    grand_t0 = time.perf_counter()

    for K in K_LIST:
        for eps_val in EPS_LIST:
            print(f"K={K}  eps={eps_val}  samples={N}")
            print(
                f"{'#':>4}  {'label':>5}  {'keep':>4}  {'unc':>3}  "
                f"{'cases':>5}  {'D_diff':>10}  {'D_zono':>10}  "
                f"{'D_early':>10}  {'t_diff':>7}  {'t_zono':>7}  "
                f"{'t_early':>8}"
            )
            print("-" * 105)

            all_diff, all_zono, all_early = [], [], []
            t_total = time.perf_counter()

            for i, (img, label) in enumerate(samples):
                with torch.no_grad():
                    x = (img - 0.1307) / 0.3081
                    x = vit.to_patch_embedding(x)
                    center = (
                        torch.cat((vit.cls_token[0], x), dim=0)
                        + vit.pos_embedding[0]
                    )

                # Importance score bounds
                with _Quiet():
                    full_zono_expr = build_zonotope_no_cat(
                        vit, img, eps_val, op_patch
                    )
                    prop._UB_CACHE.clear()
                    prop._LB_CACHE.clear()
                    score_zono_expr = op_score(full_zono_expr)
                    ub_sc, lb_sc = score_zono_expr.ublb()

                # Token classification
                definite_keep, definite_prune, uncertain = classify_topk(
                    ub_sc, lb_sc, K
                )

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
                    n_combos = comb(len(uncertain_list), K_remaining)
                    if n_combos <= MAX_CASES:
                        cases = [
                            definite_keep | set(c)
                            for c in combinations(uncertain_list, K_remaining)
                        ]
                    else:
                        mid = (ub_sc + lb_sc) / 2.0
                        ranked = sorted(
                            uncertain_list,
                            key=lambda t: mid[t].item(),
                            reverse=True,
                        )
                        cases = [definite_keep | set(ranked[:K_remaining])]

                # --- Differential baseline ---
                t0 = time.perf_counter()
                with _Quiet():
                    ub_d, lb_d = certify_diff(vit, gm_full, center, eps_val, cases)
                D_diff = D_from_ublb(ub_d, lb_d)
                t_diff = time.perf_counter() - t0

                # --- Zono-sub baseline ---
                t0 = time.perf_counter()
                with _Quiet():
                    ub_z, lb_z = certify_zono_sub(
                        vit, gm_full, center, eps_val, cases
                    )
                D_zono = D_from_ublb(ub_z, lb_z)
                t_zono = time.perf_counter() - t0

                # --- Early union + differential ---
                t0 = time.perf_counter()
                with _Quiet():
                    ub_e, lb_e = certify_early_union_diff(
                        vit, gm_attn_full, gm_tail, center, eps_val, cases
                    )
                D_early = D_from_ublb(ub_e, lb_e)
                t_early = time.perf_counter() - t0

                all_diff.append(D_diff)
                all_zono.append(D_zono)
                all_early.append(D_early)

                print(
                    f"{i+1:4d}  {label:5d}  {len(definite_keep):4d}  "
                    f"{len(uncertain):3d}  {len(cases):5d}  "
                    f"{D_diff:10.4f}  {D_zono:10.4f}  {D_early:10.4f}  "
                    f"{t_diff:6.1f}s  {t_zono:6.1f}s  {t_early:7.1f}s",
                    flush=True,
                )

            # Summary
            print("-" * 105)
            td = torch.tensor(all_diff)
            tz = torch.tensor(all_zono)
            te = torch.tensor(all_early)
            print(
                f"  {'':>30}  {'Diff':>10}  {'Zono':>10}  "
                f"{'Early':>10}  {'E/D':>6}  {'E/Z':>6}"
            )
            print(
                f"  Mean   {'':>21}  {td.mean().item():10.4f}  "
                f"{tz.mean().item():10.4f}  {te.mean().item():10.4f}  "
                f"{te.mean().item()/(td.mean().item()+1e-30):5.2f}x  "
                f"{te.mean().item()/(tz.mean().item()+1e-30):5.2f}x"
            )
            print(
                f"  Median {'':>21}  {td.median().item():10.4f}  "
                f"{tz.median().item():10.4f}  {te.median().item():10.4f}  "
                f"{te.median().item()/(td.median().item()+1e-30):5.2f}x  "
                f"{te.median().item()/(tz.median().item()+1e-30):5.2f}x"
            )
            print(
                f"  Max    {'':>21}  {td.max().item():10.4f}  "
                f"{tz.max().item():10.4f}  {te.max().item():10.4f}"
            )
            elapsed_block = time.perf_counter() - t_total
            print(f"  Block time: {elapsed_block:.0f}s\n")

    print(
        f"Grand total: {time.perf_counter() - grand_t0:.0f}s "
        f"({(time.perf_counter() - grand_t0)/60:.1f}m)"
    )


if __name__ == "__main__":
    main()