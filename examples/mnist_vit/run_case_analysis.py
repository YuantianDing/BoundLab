"""Per-class interval analysis for pruning cases.

For each test sample and each pruning case, prints the per-class output
intervals and checks whether the prediction is certifiably preserved
(i.e., the correct class has a non-overlapping margin after accounting
for the pruning difference).

Usage::

    python run_case_analysis.py
    python run_case_analysis.py --eps 0.002 --K 12 --n-samples 10
    python run_case_analysis.py --depth 3 --checkpoint mnist_transformer_3.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mnist_vit import ViT, build_mnist_vit
from pipeline import PrunedViT, ScoringModel
from token_pruning import (
    build_token_scores, build_all_kept_scores,
    classify_topk, enumerate_pruning_cases,
    build_input_zonotope, export_patch_embedding,
    export_scoring, export_pruned_vit,
)

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret


def _build_model(depth: int, checkpoint: str) -> ViT:
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


def _certified_class(ub: Tensor, lb: Tensor) -> int | None:
    """Return the certified predicted class, or None if not certifiable.

    A class is certified if its lower bound exceeds the upper bound
    of every other class.
    """
    pred = int(ub.argmax().item())
    ub_others = ub.clone()
    ub_others[pred] = float("-inf")
    margin = float(lb[pred] - ub_others.max())
    return pred if margin > 0 else None


def run_case_analysis(
    depth=1, checkpoint="mnist_transformer.pt",
    eps=0.004, K=8,
    n_samples=5, seed=0, score_layer=0, mc_samples=500,
):
    torch.manual_seed(seed)
    num_tokens, dim, num_classes = 16, 64, 10

    vit = _build_model(depth, checkpoint)
    op_patch = export_patch_embedding(vit, [1, 28, 28])
    op_score, _ = export_scoring(vit, num_tokens, dim, score_layer=score_layer)

    gm_full = export_pruned_vit(vit, set(range(num_tokens)),
                                num_tokens, dim, mask_from_layer=score_layer)
    scoring_model = ScoringModel(vit, score_layer=score_layer).eval()

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
    except Exception as e:
        g = torch.Generator().manual_seed(seed)
        samples = [(torch.rand(1, 28, 28, generator=g), -1) for _ in range(n_samples)]

    # ---- Full model bounds (no pruning) ----
    print("=" * 80)
    print(f"  Per-Case Interval Analysis — {depth}-layer ViT, eps={eps}, K={K}/{num_tokens}")
    print("=" * 80)

    total_samples = len(samples)
    certified_full = 0
    certified_pruned_union = 0
    certified_per_case_all = 0

    for sample_idx, (img, label) in enumerate(samples):
        with torch.no_grad():
            pred = int(vit(img).argmax().item())

        # Full model logit bounds
        full_zono = build_input_zonotope(vit, img, eps, op_patch)
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        full_out = zono.interpret(gm_full)(full_zono)
        full_ub, full_lb = full_out.ublb()

        cert_full = _certified_class(full_ub, full_lb)
        if cert_full is not None:
            certified_full += 1

        # Score → classify → enumerate
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        ub_sc, lb_sc = op_score(full_zono).ublb()
        dk, dp, unc = classify_topk(ub_sc, lb_sc, K)
        cases = enumerate_pruning_cases(dk, unc, K)

        print(f"\n{'─' * 80}")
        print(f"  Sample {sample_idx + 1}/{total_samples}  "
              f"label={label}  pred={pred}  "
              f"certified_full={'YES' if cert_full is not None else 'NO'}")
        print(f"  tokens: keep={len(dk)} prune={len(dp)} uncertain={len(unc)}  "
              f"cases={len(cases)}")
        print(f"{'─' * 80}")

        # ---- Full model intervals ----
        print(f"\n  Full model (no pruning) — per-class intervals:")
        print(f"  {'Class':>5}  {'[lb':>10}  {'ub]':>10}  {'width':>8}  note")
        for c in range(num_classes):
            w = full_ub[c] - full_lb[c]
            note = " ◄ pred" if c == pred else (" ◄ TRUE" if c == label and label != pred else "")
            print(f"  C_{c:>2}:  {full_lb[c]:>10.4f}  {full_ub[c]:>10.4f}  {w:>8.4f}{note}")

        # ---- Monte Carlo ground truth ----
        mc_diff_max = torch.full((num_classes,), -float("inf"))
        mc_diff_min = torch.full((num_classes,), float("inf"))
        mc_pruned_max = torch.full((num_classes,), -float("inf"))
        mc_pruned_min = torch.full((num_classes,), float("inf"))

        model_full_mc = PrunedViT(
            vit, build_all_kept_scores(num_tokens),
            mask_from_layer=score_layer, for_verification=False,
        ).eval()

        with torch.no_grad():
            for t in range(mc_samples):
                torch.manual_seed(t)
                delta = (2 * torch.rand_like(img) - 1) * eps
                img_p = img + delta
                emb_p = vit.to_patch_embedding(img_p)
                xp = torch.cat((vit.cls_token[0], emb_p), dim=0) + vit.pos_embedding[0]

                _, topk_idx = scoring_model(xp).topk(K)
                kept_mc = set(topk_idx.tolist())

                model_pruned_mc = PrunedViT(
                    vit, build_token_scores(num_tokens, kept_mc),
                    mask_from_layer=score_layer, for_verification=False,
                ).eval()

                y_full = model_full_mc(xp)
                y_pruned = model_pruned_mc(xp)
                d = y_full - y_pruned

                mc_diff_max = torch.maximum(mc_diff_max, d)
                mc_diff_min = torch.minimum(mc_diff_min, d)
                mc_pruned_max = torch.maximum(mc_pruned_max, y_pruned)
                mc_pruned_min = torch.minimum(mc_pruned_min, y_pruned)

        print(f"\n  MC ground truth ({mc_samples} pixel perturbations) — per-class diff:")
        print(f"  {'Class':>5}  {'mc_d_lb':>10}  {'mc_d_ub':>10}  {'mc_width':>10}")
        for c in range(num_classes):
            mw = mc_diff_max[c] - mc_diff_min[c]
            print(f"  C_{c:>2}:  {mc_diff_min[c]:>10.4f}  {mc_diff_max[c]:>10.4f}  {mw:>10.4f}")

        # ---- Per-case diff bounds ----
        union_diff_ub = None
        union_diff_lb = None
        case_certified = []

        for case_idx, kept in enumerate(cases):
            gm_pruned = export_pruned_vit(vit, kept, num_tokens, dim, score_layer)
            prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
            merged = diff_net(gm_full, gm_pruned)
            out = diff_interpret(merged)(full_zono)
            if isinstance(out, DiffExpr3):
                d_ub, d_lb = out.diff.ublb()
            else:
                d_ub, d_lb = (out.x - out.y).ublb()

            # Pruned model bounds derived from full bounds + diff bounds:
            #   pruned = full - diff  →  pruned ∈ [full_lb - d_ub, full_ub - d_lb]
            pruned_lb = full_lb - d_ub
            pruned_ub = full_ub - d_lb

            cert_case = _certified_class(pruned_ub, pruned_lb)
            case_certified.append(cert_case)

            if union_diff_ub is None:
                union_diff_ub = d_ub.clone()
                union_diff_lb = d_lb.clone()
            else:
                union_diff_ub = torch.maximum(union_diff_ub, d_ub)
                union_diff_lb = torch.minimum(union_diff_lb, d_lb)

            pruned_set = set(range(num_tokens)) - kept
            pruned_str = ",".join(str(p) for p in sorted(pruned_set))

            print(f"\n  Case {case_idx + 1}/{len(cases)} — "
                  f"pruned=[{pruned_str}]  "
                  f"certified={'YES (' + str(cert_case) + ')' if cert_case is not None else 'NO'}")
            print(f"  {'Class':>5}  {'diff_lb':>10}  {'diff_ub':>10}  "
                  f"{'prun_lb':>10}  {'prun_ub':>10}  {'width':>8}")
            for c in range(num_classes):
                pw = pruned_ub[c] - pruned_lb[c]
                print(f"  C_{c:>2}:  {d_lb[c]:>10.4f}  {d_ub[c]:>10.4f}  "
                      f"{pruned_lb[c]:>10.4f}  {pruned_ub[c]:>10.4f}  {pw:>8.4f}")

        # ---- Union over all cases ----
        union_pruned_lb = full_lb - union_diff_ub
        union_pruned_ub = full_ub - union_diff_lb
        cert_union = _certified_class(union_pruned_ub, union_pruned_lb)

        if cert_union is not None:
            certified_pruned_union += 1
        if all(c is not None for c in case_certified):
            certified_per_case_all += 1

        print(f"\n  Union over {len(cases)} cases — "
              f"certified={'YES (' + str(cert_union) + ')' if cert_union is not None else 'NO'}")
        print(f"  {'Class':>5}  {'diff_lb':>10}  {'diff_ub':>10}  "
              f"{'prun_lb':>10}  {'prun_ub':>10}  {'width':>8}")
        for c in range(num_classes):
            pw = union_pruned_ub[c] - union_pruned_lb[c]
            print(f"  C_{c:>2}:  {union_diff_lb[c]:>10.4f}  {union_diff_ub[c]:>10.4f}  "
                  f"{union_pruned_lb[c]:>10.4f}  {union_pruned_ub[c]:>10.4f}  {pw:>8.4f}")

        # ---- MC vs Verified comparison ----
        print(f"\n  Tightness: MC diff width vs verified diff width")
        print(f"  {'Class':>5}  {'mc_width':>10}  {'union_width':>12}  {'ratio':>7}  "
              f"{'overapprox':>10}")
        avg_mc_w = 0.0
        avg_union_w = 0.0
        for c in range(num_classes):
            mc_w = (mc_diff_max[c] - mc_diff_min[c]).item()
            union_w = (union_diff_ub[c] - union_diff_lb[c]).item()
            ratio = union_w / mc_w if mc_w > 1e-8 else float("inf")
            overapprox = union_w - mc_w
            avg_mc_w += mc_w
            avg_union_w += union_w
            print(f"  C_{c:>2}:  {mc_w:>10.4f}  {union_w:>12.4f}  {ratio:>7.1f}x  "
                  f"{overapprox:>+10.4f}")
        avg_mc_w /= num_classes
        avg_union_w /= num_classes
        avg_ratio = avg_union_w / avg_mc_w if avg_mc_w > 1e-8 else float("inf")
        print(f"  {'avg':>5}:  {avg_mc_w:>10.4f}  {avg_union_w:>12.4f}  {avg_ratio:>7.1f}x  "
              f"{avg_union_w - avg_mc_w:>+10.4f}")

    # ---- Summary ----
    print(f"\n{'=' * 80}")
    print(f"  CERTIFIED ACCURACY SUMMARY ({total_samples} samples)")
    print(f"{'=' * 80}")
    print(f"  Full model (no pruning):      "
          f"{certified_full}/{total_samples}  "
          f"({100 * certified_full / total_samples:.1f}%)")
    print(f"  Pruned (all cases certified): "
          f"{certified_per_case_all}/{total_samples}  "
          f"({100 * certified_per_case_all / total_samples:.1f}%)")
    print(f"  Pruned (union certified):     "
          f"{certified_pruned_union}/{total_samples}  "
          f"({100 * certified_pruned_union / total_samples:.1f}%)")
    if certified_full > 0:
        retained = certified_pruned_union / certified_full * 100
        print(f"\n  Certification retention: {retained:.1f}% of full-model "
              f"certifications preserved after pruning")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Per-class interval analysis for pruning cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--depth", type=int, default=1, choices=[1, 3])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eps", type=float, default=0.004)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=5, dest="n_samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc-samples", type=int, default=500, dest="mc_samples")
    ap.add_argument("--score-layer", type=int, default=0, dest="score_layer")
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = ("mnist_transformer.pt" if args.depth == 1
                           else f"mnist_transformer_{args.depth}.pt")

    run_case_analysis(**vars(args))


if __name__ == "__main__":
    main()