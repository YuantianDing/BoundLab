"""Differential token-pruning verification on MNIST ViT.

Compares three approaches for bounding ``||full(x) - pruned(x)||``:

  1. **MC** — empirical lower bound (sample perturbations, concrete top-K).
  2. **Zono-Sub** — shared zonotope through both models, subtract.
  3. **Diff (Zono3)** — differential zonotope via ``diff_net`` (tightest).

Expected ordering: ``MC ≤ Diff ≤ Zono-Sub``.

Usage::

    python run_pruning.py
    python run_pruning.py --depth 3 --checkpoint mnist_transformer_3.pt
    python run_pruning.py --eps 0.002 --K 12 --n-samples 5 --mc-samples 500
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import builtins
_real_print = builtins.print
def _filtered_print(*args, **kwargs):
    if args and "[matmul reset]" in str(args[0]):
        return
    _real_print(*args, **kwargs)
builtins.print = _filtered_print

import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mnist_vit import ViT, build_mnist_vit
from pipeline import ScoringModel, PrunedViT
from token_pruning import (
    build_token_scores, build_all_kept_scores,
    classify_topk, enumerate_pruning_cases,
    build_input_zonotope, export_patch_embedding,
    export_scoring, export_pruned_vit,
)

import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_model(depth: int, checkpoint: str) -> ViT:
    """Build the MNIST ViT with the given depth and checkpoint."""
    if depth == 1:
        return build_mnist_vit(checkpoint)
    model = ViT(
        image_size=28, patch_size=7, num_classes=10, channels=1,
        dim=64, depth=depth, heads=4, mlp_dim=128,
        layer_norm_type="no_var", pool="cls", dim_head=64,
    )
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(os.path.dirname(__file__), checkpoint)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.eval()


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo_bound(
    vit, scoring_model, img, eps, K,
    num_tokens, dim, n_mc=1000, mask_from_layer=0,
):
    """Empirical lower bound on max|full(x) - pruned(x)| over pixel perturbations.

    Perturbs the image in pixel space (L∞ ball of radius eps), propagates
    through the patch embedding, then compares full vs pruned outputs.
    This matches the perturbation set that the verification bounds.
    """
    full_scores = build_all_kept_scores(num_tokens)
    model_full = PrunedViT(vit, full_scores, mask_from_layer=mask_from_layer).eval()

    mc_max = 0.0
    with torch.no_grad():
        for t in range(n_mc):
            torch.manual_seed(t)
            delta = (2 * torch.rand_like(img) - 1) * eps
            img_p = img + delta

            # Full pipeline: pixel → patch_embed → cat CLS → add pos_emb
            emb_p = vit.to_patch_embedding(img_p)
            xp = torch.cat((vit.cls_token[0], emb_p), dim=0) + vit.pos_embedding[0]

            _, topk_idx = scoring_model(xp).topk(K)
            kept = set(topk_idx.tolist())

            pruned_scores = build_token_scores(num_tokens, kept)
            model_pruned = PrunedViT(
                vit, pruned_scores, mask_from_layer=mask_from_layer,
            ).eval()

            diff = model_full(xp) - model_pruned(xp)
            mc_max = max(mc_max, diff.abs().max().item())

    return mc_max


# ---------------------------------------------------------------------------
# Verification methods (per case)
# ---------------------------------------------------------------------------

def _zono_sub_case(gm_full, gm_pruned, full_zono):
    """Shared zonotope, subtract."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    d = zono.interpret(gm_full)(full_zono) - zono.interpret(gm_pruned)(full_zono)
    return d.ublb()


def _diff_case(gm_full, gm_pruned, full_zono):
    """Differential zonotope via diff_net."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    merged = diff_net(gm_full, gm_pruned)
    out = diff_interpret(merged)(full_zono)
    if isinstance(out, DiffExpr3):
        return out.diff.ublb()
    return (out.x - out.y).ublb()


def _run_method(method_fn, gm_full, cases, vit, num_tokens, dim,
                full_zono, mask_from_layer=0):
    """Run method over all case-split branches, union bounds."""
    best_ub = best_lb = None
    t0 = time.perf_counter()

    for kept in cases:
        gm_pruned = export_pruned_vit(vit, kept, num_tokens, dim, mask_from_layer)
        d_ub, d_lb = method_fn(gm_full, gm_pruned, full_zono)
        if best_ub is None:
            best_ub, best_lb = d_ub.clone(), d_lb.clone()
        else:
            best_ub = torch.maximum(best_ub, d_ub)
            best_lb = torch.minimum(best_lb, d_lb)

    elapsed = time.perf_counter() - t0
    bound = max(best_ub.abs().max().item(), best_lb.abs().max().item())
    width = (best_ub - best_lb).mean().item()
    return bound, width, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    depth=1, checkpoint="mnist_transformer.pt",
    eps=0.004, K=8,
    n_samples=3, mc_samples=500, seed=0,
    debug=False, score_layer=0,
):
    torch.manual_seed(seed)
    num_tokens, dim = 16, 64

    vit = _build_model(depth, checkpoint)

    print(f"[setup] exporting sub-models ({depth}-layer) ...", flush=True)
    t0 = time.time()

    op_patch = export_patch_embedding(vit, [1, 28, 28])
    op_score, scoring_model = export_scoring(vit, num_tokens, dim,
                                              score_layer=score_layer)

    gm_full = export_pruned_vit(vit, set(range(num_tokens)),
                                num_tokens, dim, mask_from_layer=score_layer)

    print(f"[setup] done ({time.time() - t0:.1f}s)\n")

    # ---- Load data ----
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(
            str(_HERE / "mnist_data"), train=False, download=True,
            transform=transforms.ToTensor(),
        )
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(ds), generator=g)[:n_samples].tolist()
        samples = [(ds[i][0], int(ds[i][1])) for i in indices]
        print(f"[data] loaded {len(samples)} MNIST test samples\n")
    except Exception as e:
        print(f"[data] MNIST unavailable ({e}); using synthetic inputs\n")
        g = torch.Generator().manual_seed(seed)
        samples = [(torch.rand(1, 28, 28, generator=g), -1) for _ in range(n_samples)]

    # ---- Header ----
    hdr = (f"{'#':>3}  {'lbl':>3} {'pred':>4}  "
           f"{'keep':>4}/{'prn':>3}/{'unc':>3}  {'cases':>5}  "
           f"{'MC':>10}  {'Zono-Sub':>10}  {'Diff':>10}  "
           f"{'t_MC':>5} {'t_ZS':>5} {'t_D':>5}")
    print(hdr)
    print("-" * len(hdr))

    all_mc, all_zs, all_diff = [], [], []

    for i, (img, label) in enumerate(samples):
        with torch.no_grad():
            pred = int(vit(img).argmax().item())

        # ---- Classify tokens ----
        full_zono = build_input_zonotope(vit, img, eps, op_patch)
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        ub_sc, lb_sc = op_score(full_zono).ublb()
        definite_keep, definite_prune, uncertain = classify_topk(ub_sc, lb_sc, K)
        cases = enumerate_pruning_cases(definite_keep, uncertain, K)

        if debug:
            width = ub_sc - lb_sc
            ranked = torch.argsort(-(ub_sc + lb_sc) / 2)
            _real_print(f"    score bounds (sorted by midpoint, K={K}):")
            for r, idx in enumerate(ranked):
                tag = "KEEP" if idx.item() in definite_keep else (
                      "PRUNE" if idx.item() in definite_prune else "UNC")
                sep = " <-- K boundary" if r == K - 1 else ""
                _real_print(f"      patch {idx.item():2d}: [{lb_sc[idx]:.4f}, {ub_sc[idx]:.4f}]"
                            f"  width={width[idx]:.4f}  {tag}{sep}")

        # ---- Monte Carlo ----
        t_mc = time.perf_counter()
        mc_bound = monte_carlo_bound(
            vit, scoring_model, img, eps, K,
            num_tokens, dim, mc_samples, mask_from_layer=score_layer,
        )
        t_mc = time.perf_counter() - t_mc
        all_mc.append(mc_bound)

        # ---- Zono-Sub ----
        zs_bound, zs_width, t_zs = _run_method(
            _zono_sub_case, gm_full, cases,
            vit, num_tokens, dim, full_zono, mask_from_layer=score_layer,
        )
        all_zs.append(zs_bound)

        # ---- Differential ----
        diff_bound, diff_width, t_diff = _run_method(
            _diff_case, gm_full, cases,
            vit, num_tokens, dim, full_zono, mask_from_layer=score_layer,
        )
        all_diff.append(diff_bound)

        lbl = f"{label:>3}" if label >= 0 else "  -"
        print(f"{i+1:>3}  {lbl} {pred:>4}  "
              f"{len(definite_keep):>4}/{len(definite_prune):>3}/{len(uncertain):>3}  "
              f"{len(cases):>5}  "
              f"{mc_bound:>10.6f}  {zs_bound:>10.4f}  {diff_bound:>10.4f}  "
              f"{t_mc:>5.1f} {t_zs:>5.1f} {t_diff:>5.1f}")

    # ---- Summary ----
    n = len(samples)
    avg_mc = sum(all_mc) / n
    avg_zs = sum(all_zs) / n
    avg_diff = sum(all_diff) / n

    print()
    print("=" * 72)
    print(f"  Differential Pruning Verification — MNIST ViT ({depth} layer{'s' if depth > 1 else ''})")
    print(f"  checkpoint: {checkpoint}")
    print(f"  eps={eps}, K={K}/{num_tokens}, score_layer={score_layer}, MC={mc_samples}")
    print("=" * 72)
    print()
    print(f"  {'Method':<20} {'Avg Bound':>10}")
    print(f"  {'-'*35}")
    print(f"  {'MC (empirical)':<20} {avg_mc:>10.6f}  (lower bound)")
    print(f"  {'Zono-Sub':<20} {avg_zs:>10.4f}  (sound upper bound)")
    print(f"  {'Diff (ours)':<20} {avg_diff:>10.4f}  (sound upper bound)")

    if avg_zs > 0 and avg_diff > 0:
        if avg_diff < avg_zs:
            pct = (1 - avg_diff / avg_zs) * 100
            print(f"\n  Diff is {pct:.1f}% tighter than Zono-Sub on average")
        else:
            pct = (1 - avg_zs / avg_diff) * 100
            print(f"\n  Zono-Sub is {pct:.1f}% tighter than Diff on average")

    print(f"\n  Soundness: MC <= each sound bound")
    all_sound = True
    for i in range(n):
        ok_diff = all_mc[i] <= all_diff[i] + 1e-6
        ok_zs = all_mc[i] <= all_zs[i] + 1e-6
        ok = ok_diff and ok_zs
        if not ok:
            all_sound = False
        tag = "OK" if ok else "FAIL"
        tighter = "Diff" if all_diff[i] <= all_zs[i] else "ZS"
        print(f"    sample {i+1}: MC={all_mc[i]:.6f}  "
              f"Diff={all_diff[i]:.4f}  ZS={all_zs[i]:.4f}  "
              f"[{tag}, {tighter} tighter]")

    if all_sound:
        print(f"\n  All {n} samples sound.")
    else:
        print(f"\n  WARNING: soundness violation!")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Pruning verification: MC vs Zono-Sub vs Differential.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--depth", type=int, default=1, choices=[1, 3],
                    help="Transformer depth (1 or 3)")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint path (default: mnist_transformer.pt for depth=1, "
                         "mnist_transformer_3.pt for depth=3)")
    ap.add_argument("--eps", type=float, default=0.004)
    ap.add_argument("--K", type=int, default=8,
                    help="Number of patch tokens to keep (out of 16)")
    ap.add_argument("--n-samples", type=int, default=3, dest="n_samples")
    ap.add_argument("--mc-samples", type=int, default=500, dest="mc_samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true",
                    help="Print score bounds per token")
    ap.add_argument("--score-layer", type=int, default=0, dest="score_layer",
                    help="Which transformer layer's CLS attention to use for scoring")
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = ("mnist_transformer.pt" if args.depth == 1
                           else f"mnist_transformer_{args.depth}.pt")

    run(depth=args.depth, **{k: v for k, v in vars(args).items()
                             if k not in ("depth",)})


if __name__ == "__main__":
    main()