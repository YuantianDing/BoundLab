"""Differential token-pruning verification on the MNIST ViT (vit/ pipeline).

Compares five bounds on ``||full(x) - pruned(x)||_inf`` over an L-inf ball of
radius ``eps`` around each test image:

    MC             — empirical lower bound (sample perturbations, concrete
                     masked attention against the clean ViT).
    zono           — ``(out.x - out.y).ublb()`` from the DiffExpr3 returned by
                     ``boundlab.diff.zono3.interpret``; subtraction over the
                     shared zonotope (no VeryDiff exploitation).
    zono3          — ``out.diff.ublb()`` from the same DiffExpr3; uses the
                     VeryDiff-style linearisation in
                     ``boundlab.diff.zono3.default``.
    zonohex        — same as zono3 but the interpreter chains
                     ``expr3_to_expr2`` after each handler via
                     ``boundlab.diff.zonohex.interpret`` (ZonoHexGate
                     re-relaxation).
    zono3/gradlin  — zono3.interpret with the gradient-descent-tightened
                     linearisers from ``boundlab.diff.zono3.gradlin`` swapped
                     in for Exp/Tanh/Reciprocal/Softmax.

Usage::

    python run_pruning.py
    python run_pruning.py --depth 3 --eps 0.002 --K 12 --n-samples 5
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.expr as _expr  # noqa: F401  (ensures core is initialised before prop)
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.diff.zono3 import interpret as zono3_interpret, linearizer_to_hander
from boundlab.diff.zono3 import gradlin as _gradlin
from boundlab.diff.zonohex import interpret as zonohex_interpret
from boundlab.expr._affine import ConstVal
from boundlab.expr._var import LpEpsilon
from boundlab.interp import Interpreter

from mnist_vit import mnist_vit as mnist_vit_clean
from vit_pruning_diff import mnist_vit_pruning, scoring_model


# ---------------------------------------------------------------------------
# Token classification (same shape as mnist_vit/token_pruning.py)
# ---------------------------------------------------------------------------

def classify_topk(ub_scores: torch.Tensor, lb_scores: torch.Tensor, K: int):
    """Partition N patch tokens into definite-keep / definite-prune / uncertain."""
    N = len(ub_scores)
    n_prune = N - K
    wins = torch.zeros(N, dtype=torch.long)
    losses = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if lb_scores[i] > ub_scores[j]:
                wins[i] += 1
            if ub_scores[i] < lb_scores[j]:
                losses[i] += 1
    definite_keep = {i for i in range(N) if wins[i] >= n_prune}
    definite_prune = {i for i in range(N) if losses[i] >= K}
    uncertain = set(range(N)) - definite_keep - definite_prune
    return definite_keep, definite_prune, uncertain


def enumerate_cases(definite_keep: set[int], uncertain: set[int], K: int) -> list[set[int]]:
    K_rem = K - len(definite_keep)
    if K_rem < 0:
        return [definite_keep.copy()]
    if K_rem >= len(uncertain):
        return [definite_keep | uncertain]
    if K_rem == 0:
        return [definite_keep.copy()]
    return [definite_keep | set(c)
            for c in combinations(sorted(uncertain), K_rem)]


def build_token_mask(num_tokens: int, kept: set[int], magnitude: float = 100.0) -> torch.Tensor:
    """``[num_tokens+1]`` tensor: ``+magnitude`` for kept (incl. CLS at index 0)."""
    m = torch.full((num_tokens + 1,), -magnitude)
    m[0] = magnitude
    for p in kept:
        m[p + 1] = magnitude
    return m


# ---------------------------------------------------------------------------
# MC — concrete masked forward against the clean ViT
# ---------------------------------------------------------------------------

def _masked_forward(vit, img: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Run ``vit`` (clean ``mnist_vit.ViT``) with the y-branch semantics of
    ``softmax_pruning`` from ``boundlab.diff.op``: per-row attention is

        attn[h, q, k] = exp(dots[h, q, k])
                      / sum_{k'} h(token_mask[k']) * exp(dots[h, q, k'])

    i.e. pruned keys appear in the numerator but not the denominator. This is
    what the diff zonotope tracks as the second network branch, so MC must
    mirror it to give a true empirical lower bound on ``||x - y||``.
    """
    x = vit.to_patch_embedding(img)
    cls = vit.cls_token[0]
    x = torch.cat((cls, x), dim=0) + vit.pos_embedding[0]
    keep_k = (token_mask > 0).to(img.dtype).view(1, 1, -1)  # (1, 1, n)

    for attn_block, ff_block in vit.transformer.layers:
        prenorm = attn_block.fn
        attn = prenorm.fn
        residual = x
        xn = prenorm.norm(x)
        n = xn.shape[0]
        h, d = attn.heads, attn.dim_head
        q = attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)
        k = attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
        v = attn.to_v(xn).reshape(n, h, d).permute(1, 0, 2)
        dots = (q @ k.transpose(-2, -1)) * attn.scale

        # Denominator-only masked softmax (softmax_pruning y-branch).
        m = dots.amax(dim=-1, keepdim=True)
        exp_dots = (dots - m).exp()
        denom = (exp_dots * keep_k).sum(dim=-1, keepdim=True)
        attn_w = exp_dots / denom

        out = (attn_w @ v).permute(1, 0, 2).reshape(n, h * d)
        out = attn.to_out(out)
        x = residual + out
        x = ff_block(x)

    x = x.mean(0) if vit.pool == "mean" else x[0]
    return vit.mlp_head(x)


def monte_carlo(vit, img: torch.Tensor, eps: float,
                full_mask: torch.Tensor, prn_mask: torch.Tensor,
                n_mc: int = 500) -> float:
    max_diff = 0.0
    with torch.no_grad():
        for t in range(n_mc):
            torch.manual_seed(t)
            img_p = img + (2 * torch.rand_like(img) - 1) * eps
            diff = _masked_forward(vit, img_p, full_mask) - _masked_forward(vit, img_p, prn_mask)
            max_diff = max(max_diff, diff.abs().max().item())
    return max_diff


# ---------------------------------------------------------------------------
# Per-case bound extraction
# ---------------------------------------------------------------------------

def _diff2_bounds(out):
    """``(x - y).ublb()`` for DiffExpr2/3 outputs."""
    if isinstance(out, (DiffExpr2, DiffExpr3)):
        return (out.x - out.y).ublb()
    raise TypeError(f"Unexpected interpreter output: {type(out)}")


def _diff3_bounds(out):
    """``diff.ublb()`` for DiffExpr3 outputs (falls back to ``x - y`` for DiffExpr2)."""
    if isinstance(out, DiffExpr3):
        return out.diff.ublb()
    if isinstance(out, DiffExpr2):
        return (out.x - out.y).ublb()
    raise TypeError(f"Unexpected interpreter output: {type(out)}")


def case_bounds(op_zono3, op_zonohex, op_gradlin,
                img: torch.Tensor, eps: float, mask: torch.Tensor):
    """Run the three differential interpreters for a fixed ``mask`` and return
    bounds for ``zono``, ``zono3``, ``zonohex``, ``zono3/gradlin``.
    """
    img_zono_factory = lambda: ConstVal(img) + eps * LpEpsilon(list(img.shape))

    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    out = op_zono3(img_zono_factory(), ConstVal(mask))
    zono_ub, zono_lb = _diff2_bounds(out)
    zono3_ub, zono3_lb = _diff3_bounds(out)

    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    out_h = op_zonohex(img_zono_factory(), ConstVal(mask))
    hex_ub, hex_lb = _diff2_bounds(out_h)

    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    out_g = op_gradlin(img_zono_factory(), ConstVal(mask))
    grad_ub, grad_lb = _diff3_bounds(out_g)

    return {
        "zono":          (zono_ub, zono_lb),
        "zono3":         (zono3_ub, zono3_lb),
        "zonohex":       (hex_ub, hex_lb),
        "zono3/gradlin": (grad_ub, grad_lb),
    }


def _build_gradlin_interpreter():
    """Fresh copy of zono3.interpret with gradlin handlers registered.

    ``Interpreter(zono3_interpret)`` copies each FnList (shallow list copy),
    so :func:`register_all`'s in-place additions don't leak into the global
    ``zono3.interpret``.
    """
    interp = Interpreter(zono3_interpret)
    _gradlin.register_all(interp, linearizer_to_hander)
    return interp


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_samples(n: int, data_dir: Path, seed: int):
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(str(data_dir), train=False, download=True,
                            transform=transforms.ToTensor())
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(ds), generator=g)[:n].tolist()
        samples = [(ds[i][0], int(ds[i][1])) for i in indices]
        print(f"[data] loaded {len(samples)} MNIST test samples\n")
        return samples
    except Exception as e:
        print(f"[data] MNIST unavailable ({e}); using synthetic inputs\n")
        g = torch.Generator().manual_seed(seed)
        return [(torch.rand(1, 28, 28, generator=g), -1) for _ in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(depth: int = 1, eps: float = 0.004, K: int = 12,
        n_samples: int = 3, mc_samples: int = 500, seed: int = 0):
    torch.manual_seed(seed)
    num_tokens = 16

    print(f"[setup] building / loading ONNX (depth={depth}) ... K={K}", flush=True)
    t0 = time.time()
    pruning_path = mnist_vit_pruning(depth=depth)
    score_path = scoring_model(depth=depth)
    op_zono3 = zono3_interpret(str(pruning_path))
    op_zonohex = zonohex_interpret(str(pruning_path))
    op_gradlin = _build_gradlin_interpreter()(str(pruning_path))
    op_score = zono.interpret(str(score_path))
    vit_clean = mnist_vit_clean(depth=depth)
    print(f"[setup] done in {time.time() - t0:.1f}s\n")

    samples = load_samples(n_samples, _HERE / "mnist_data", seed)

    methods = ("zono", "zono3", "zonohex", "zono3/gradlin")

    hdr = (f"{'#':>3}  {'lbl':>3} {'pred':>4}  "
           f"{'keep':>4}/{'prn':>3}/{'unc':>3}  {'cases':>5}  "
           f"{'MC':>10}  "
           + "  ".join(f"{m:>13}" for m in methods)
           + f"  {'t_MC':>6} {'t_v':>6}")
    print(hdr)
    print("-" * len(hdr))

    all_mc: list[float] = []
    all_bounds: dict[str, list[float]] = {m: [] for m in methods}
    full_mask = build_token_mask(num_tokens, set(range(num_tokens)))

    for i, (img, label) in enumerate(samples):
        with torch.no_grad():
            pred = int(vit_clean(img).argmax().item())

        # --- Score bounds via plain zono on the sliced scoring graph ---
        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()
        img_zono = ConstVal(img) + eps * LpEpsilon(list(img.shape))
        score_expr = op_score(img_zono)              # shape [heads, N+1, N+1]
        ub_mul, lb_mul = score_expr.ublb()
        ub_scores = ub_mul[:, 0, 1:].mean(0)
        lb_scores = lb_mul[:, 0, 1:].mean(0)

        d_keep, d_prn, unc = classify_topk(ub_scores, lb_scores, K)
        cases = enumerate_cases(d_keep, unc, K)

        # --- Per-case ---
        t_mc_total = 0.0
        t_v_total = 0.0
        mc_best = 0.0
        best: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {m: None for m in methods}

        for kept in cases:
            mask = build_token_mask(num_tokens, kept)

            t = time.perf_counter()
            mc = monte_carlo(vit_clean, img, eps, full_mask, mask, n_mc=mc_samples)
            t_mc_total += time.perf_counter() - t
            mc_best = max(mc_best, mc)

            t = time.perf_counter()
            cb = case_bounds(op_zono3, op_zonohex, op_gradlin, img, eps, mask)
            t_v_total += time.perf_counter() - t

            for m in methods:
                ub, lb = cb[m]
                if best[m] is None:
                    best[m] = (ub.clone(), lb.clone())
                else:
                    cur_ub, cur_lb = best[m]
                    best[m] = (torch.maximum(cur_ub, ub), torch.minimum(cur_lb, lb))

        all_mc.append(mc_best)
        bounds = {}
        for m in methods:
            ub, lb = best[m]
            bounds[m] = max(ub.abs().max().item(), lb.abs().max().item())
            all_bounds[m].append(bounds[m])

        lbl = f"{label:>3}" if label >= 0 else "  -"
        print(f"{i+1:>3}  {lbl} {pred:>4}  "
              f"{len(d_keep):>4}/{len(d_prn):>3}/{len(unc):>3}  "
              f"{len(cases):>5}  "
              f"{mc_best:>10.6f}  "
              + "  ".join(f"{bounds[m]:>13.4f}" for m in methods)
              + f"  {t_mc_total:>6.1f} {t_v_total:>6.1f}")

    # --- Summary ---
    n = len(samples)
    avg_mc = sum(all_mc) / n
    avg = {m: sum(all_bounds[m]) / n for m in methods}
    print()
    print("=" * 78)
    print(f"  Differential Pruning Verification — MNIST ViT (depth={depth})")
    print(f"  eps={eps}, K={K}/{num_tokens}, MC={mc_samples}, n={n}")
    print("=" * 78)
    print(f"  {'MC (empirical)':<20} {avg_mc:>10.6f}  (lower bound)")
    for m in methods:
        print(f"  {m:<20} {avg[m]:>10.4f}  (sound upper bound)")

    # Tightest sound bound by method.
    tightest = min(methods, key=lambda m: avg[m])
    print(f"\n  Tightest method on average: {tightest}")

    print(f"\n  Soundness: MC <= each sound bound")
    all_sound = True
    for i in range(n):
        per_sample = {m: all_bounds[m][i] for m in methods}
        ok = all(all_mc[i] <= per_sample[m] + 1e-6 for m in methods)
        if not ok:
            all_sound = False
        best_m = min(methods, key=lambda m: per_sample[m])
        tag = "OK" if ok else "FAIL"
        details = "  ".join(f"{m}={per_sample[m]:.4f}" for m in methods)
        print(f"    sample {i+1}: MC={all_mc[i]:.6f}  {details}  [{tag}, {best_m} tightest]")
    print(f"\n  {'All ' + str(n) + ' samples sound.' if all_sound else 'WARNING: soundness violation!'}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Pruning verification: MC vs zono / zono3 / zonohex / zono3+gradlin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--depth", type=int, default=1, choices=[1, 3])
    ap.add_argument("--eps", type=float, default=0.004)
    ap.add_argument("--K", type=int, default=14, help="Top-K patches to keep (of 16)")
    ap.add_argument("--n-samples", type=int, default=3, dest="n_samples")
    ap.add_argument("--mc-samples", type=int, default=500, dest="mc_samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
