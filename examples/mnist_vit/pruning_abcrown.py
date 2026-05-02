"""Differential verification of pruned ViT using alpha-beta-CROWN / auto_LiRPA.

Bounds ||model(x) - model_pruned(x)|| under L∞ perturbation using independent
auto_LiRPA bounds on the full and pruned models, then subtracting:
  - abcrown: CROWN on each model (IBP fallback — CROWN fails on attention due
             to BoundReduceMax not implementing bound_backward for perturbed indexes)

Requires alpha-beta-CROWN cloned at ../abcrown (relative to the project root)
or the ABCROWN_DIR environment variable pointing to it.

Usage::

    python pruning_abcrown.py --checkpoint mnist_transformer.pt --eps 0.002 --K 8
    python pruning_abcrown.py --eps 0.004 --K 4 --n-samples 10
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn, Tensor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.expr  # resolve circular import before boundlab.prop
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

from mnist_vit import build_mnist_vit
from certify import PatchifyStage
from certify_pruned import ScoringModel, build_zonotope_no_cat, classify_topk
from pruning_zono import MaskedModel, load_test_samples


# ---------------------------------------------------------------------------
# alpha-beta-CROWN / auto_LiRPA helpers
# ---------------------------------------------------------------------------

_ABCROWN_TMP = _HERE / "_abcrown_tmp"
_MINIMAL_CONFIG = "general:\n  device: cpu\n  conv_mode: matrix\n"


def find_abcrown_script() -> Path | None:
    project_root = _HERE.parent.parent
    candidates: list[Path] = []
    if env := os.environ.get("ABCROWN_DIR"):
        candidates.append(Path(env).resolve())
    candidates.append((project_root.parent / "abcrown").resolve())
    candidates.append((project_root / "compare" / "abcrown" / "alpha-beta-CROWN").resolve())
    for root in candidates:
        script = root / "complete_verifier" / "abcrown.py"
        if script.is_file():
            return script
    return None


def _abcrown_python() -> str:
    return os.environ.get("ABCROWN_PYTHON", sys.executable)


def _write_vnnlib_bound_query(
    path: Path, center: Tensor, eps: float,
    num_classes: int, target_class: int, lower: bool,
) -> None:
    """Write a VNN-LIB querying lb (lower=True) or ub (lower=False) of one logit."""
    flat = center.flatten().tolist()
    lines = [f"; bound query class={target_class} lower={lower} eps={eps}"]
    for i in range(len(flat)):
        lines.append(f"(declare-const X_{i} Real)")
    for j in range(num_classes):
        lines.append(f"(declare-const Y_{j} Real)")
    for i, v in enumerate(flat):
        lines.append(f"(assert (<= X_{i} {v + eps:.8f}))")
        lines.append(f"(assert (>= X_{i} {v - eps:.8f}))")
    if lower:
        lines.append(f"(assert (>= Y_{target_class} 0))")
    else:
        lines.append(f"(assert (<= Y_{target_class} 0))")
    path.write_text("\n".join(lines) + "\n")


def _extract_lb_from_pickle(pkl_path: Path) -> float:
    with pkl_path.open("rb") as f:
        out = pickle.load(f)
    for key in ("init_alpha_crown", "init_crown_bounds", "refined_lb"):
        if out.get(key) is not None:
            return float(torch.as_tensor(out[key]).reshape(-1)[0].item())
    raise KeyError(f"No bound tensor in {pkl_path}; keys={list(out.keys())}")


def _abcrown_query_lb(
    abcrown_script: Path, onnx_path: Path, config_path: Path,
    center: Tensor, eps: float, num_classes: int,
    input_shape: list[int], tmp_dir: Path, prefix: str, timeout: int = 120,
) -> Tensor:
    """Query lower bounds for all classes via abcrown subprocess (lower=True only).

    Only lower=True queries are used because abcrown's backward pass has a shape
    mismatch for lower=False queries on attention models.  To get upper bounds,
    call this function on a negated-output ONNX and negate the result.
    """
    lb = torch.empty(num_classes)
    cmd_base = [
        _abcrown_python(), str(abcrown_script),
        "--config", str(config_path),
        "--onnx_path", str(onnx_path),
        "--device", "cpu",
        "--input_shape", "-1", *[str(d) for d in input_shape],
        "--timeout", str(timeout),
        "--complete_verifier", "skip",
        "--pgd_order", "skip",
        "--save_output",
    ]
    for cls in range(num_classes):
        spec = tmp_dir / f"{prefix}_cls{cls}.vnnlib"
        out  = tmp_dir / f"{prefix}_cls{cls}.pkl"
        _write_vnnlib_bound_query(spec, center, eps, num_classes, cls, lower=True)
        proc = subprocess.run(
            cmd_base + ["--vnnlib_path", str(spec), "--output_file", str(out)],
            capture_output=True, text=True, timeout=timeout + 30,
            cwd=abcrown_script.parent.parent,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"abcrown failed prefix={prefix} cls={cls}: {proc.stderr[-400:]}")
        lb[cls] = _extract_lb_from_pickle(out)
    return lb


class BatchedMaskedModel(nn.Module):
    """Batched wrapper around MaskedModel for auto_LiRPA (needs a batch dimension)."""

    def __init__(self, model: MaskedModel):
        super().__init__()
        self.m = model

    def forward(self, x: Tensor) -> Tensor:  # (B, N, D) -> (B, num_classes)
        m = self.m
        B, N, _ = x.shape
        h, d = m.heads, m.dim_head
        x = x * m.mask
        residual = x
        xn = m.attn_norm(x)
        q = m.attn.to_q(xn).reshape(B, N, h, d).permute(0, 2, 1, 3)
        k = m.attn.to_k(xn).reshape(B, N, h, d).permute(0, 2, 1, 3)
        v = m.attn.to_v(xn).reshape(B, N, h, d).permute(0, 2, 1, 3)
        scores = (q @ k.transpose(-2, -1)) * m.scale
        # Use exp/sum instead of softmax: avoids BoundReduceMax which CROWN
        # cannot propagate backward through when indices are perturbed.
        e = scores.exp()
        attn_w = e / e.sum(dim=-1, keepdim=True)
        out = (attn_w @ v).permute(0, 2, 1, 3).reshape(B, N, h * d)
        out = m.attn.to_out(out)
        x = residual + out
        x = m.ff_block(x)
        # narrow+squeeze avoids a Gather ONNX node that onnx2pytorch can't trace
        x = x.mean(dim=1) if m.pool == "mean" else x.narrow(1, 0, 1).squeeze(1)
        return m.mlp_head(x)


def _autolirpa_get_bounds(bm: nn.Module, center: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """Compute per-class (lb, ub) via auto_LiRPA: CROWN with IBP fallback.

    CROWN fails on attention models because BoundReduceMax (from softmax numerical
    stability decomposition) doesn't implement bound_backward for perturbed indexes.
    IBP is looser but always succeeds.
    """
    abcrown_script = find_abcrown_script()
    if abcrown_script is not None:
        verifier_dir = str(abcrown_script.parent)
        if verifier_dir not in sys.path:
            sys.path.insert(0, verifier_dir)
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    dummy = center.unsqueeze(0)  # (1, N, D)
    bnd = BoundedModule(bm, dummy, device="cpu")
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    x = BoundedTensor(dummy, ptb)
    try:
        lb, ub = bnd.compute_bounds(x=(x,), method="CROWN")
    except NotImplementedError:
        lb, ub = bnd.compute_bounds(x=(x,), method="IBP")
    return lb.squeeze(0).detach(), ub.squeeze(0).detach()


def certify_abcrown(
    bm_full: BatchedMaskedModel,
    bm_pruned: BatchedMaskedModel,
    center: Tensor,
    eps: float,
    num_classes: int = 10,
) -> tuple[Tensor, Tensor, dict]:
    """Bound difference via auto_LiRPA: independent bounds on each model, then subtract."""
    lb1, ub1 = _autolirpa_get_bounds(bm_full, center, eps)
    lb2, ub2 = _autolirpa_get_bounds(bm_pruned, center, eps)
    d_ub = ub1 - lb2
    d_lb = lb1 - ub2
    return d_ub, d_lb, {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Differential verification (abcrown/auto_LiRPA): full vs pruned ViT.",
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

    # auto_LiRPA availability check
    abcrown_script = find_abcrown_script()
    if abcrown_script is not None:
        _verifier_dir = str(abcrown_script.parent)
        if _verifier_dir not in sys.path:
            sys.path.insert(0, _verifier_dir)
    try:
        import auto_LiRPA as _al  # noqa: F401
        _have_autolirpa = True
    except ImportError:
        _have_autolirpa = False
        print("[warn] auto_LiRPA not found; set ABCROWN_DIR or clone into ../abcrown", flush=True)

    print("=" * 75)
    print(f"  Differential Verification (abcrown): full vs top-{args.K} pruned ViT")
    print("=" * 75)
    print(f"  eps={args.eps}, K={args.K}, MC={args.mc_samples}")
    print()

    all_results: list[tuple[float, float, float]] = []
    all_mc = []

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

        mask_full = torch.ones(17, 64)

        mc_max = 0.0
        model_full_mc = MaskedModel(vit, mask_full).eval()
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

        if _have_autolirpa:
            t0 = time.perf_counter()
            bm_full = BatchedMaskedModel(MaskedModel(vit, mask_full).eval())
            best_d_ub = None
            best_d_lb = None
            try:
                for case_kept in cases:
                    mask_pruned = torch.zeros(17, 64)
                    mask_pruned[0] = 1.0
                    for p in case_kept:
                        mask_pruned[p + 1] = 1.0
                    bm_pruned = BatchedMaskedModel(MaskedModel(vit, mask_pruned).eval())
                    d_ub, d_lb, _ = certify_abcrown(bm_full, bm_pruned, center, args.eps)
                    if best_d_ub is None:
                        best_d_ub = d_ub.clone()
                        best_d_lb = d_lb.clone()
                    else:
                        best_d_ub = torch.maximum(best_d_ub, d_ub)
                        best_d_lb = torch.minimum(best_d_lb, d_lb)
                elapsed = time.perf_counter() - t0
                bound = max(best_d_ub.abs().max().item(), best_d_lb.abs().max().item())
                width = (best_d_ub - best_d_lb).mean().item()
                all_results.append((bound, elapsed, width))
                print(f"    {'abcrown':<15} bound={bound:.4f}  width={width:.4f}  time={elapsed:.1f}s")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"    {'abcrown':<15} FAILED: {e} ({elapsed:.1f}s)")

        print(f"    {'MC':<15} max|diff|={mc_max:.6f}")

    print()
    print("=" * 75)
    print(f"  SUMMARY: {len(samples)} samples, eps={args.eps}, K={args.K}")
    print("=" * 75)
    avg_mc = sum(all_mc) / len(all_mc)
    print(f"  {'Method':<15} {'Avg Bound':>10} {'Avg Width':>10} {'Avg Time':>9}")
    print(f"  {'-'*48}")
    print(f"  {'MC truth':<15} {avg_mc:>10.4f}")
    if all_results:
        avg_b = sum(b for b, _, _ in all_results) / len(all_results)
        avg_w = sum(w for _, _, w in all_results) / len(all_results)
        avg_t = sum(t for _, t, _ in all_results) / len(all_results)
        print(f"  {'abcrown':<15} {avg_b:>10.4f} {avg_w:>10.4f} {avg_t:>8.1f}s")
    else:
        print(f"  {'abcrown':<15} {'(no data)':>10}")


if __name__ == "__main__":
    main()
