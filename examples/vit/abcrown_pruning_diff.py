"""auto_LiRPA / α,β-CROWN baseline for differential pruning verification.

For each test image (and each kept-set returned by :mod:`veri_pruning_diff`'s
case enumeration), this script computes per-class bounds on:

    f_full(img)     — the clean ViT with no token pruning,
    f_pruned(img)   — the same ViT with a concrete y-branch attention mask
                      (denominator-only masked softmax, matching the
                      ``softmax_pruning`` semantics used by zono3).

Bounds are obtained via :mod:`auto_LiRPA` (the engine sitting under
α,β-CROWN at ``../abcrown``). The differential upper bound on
``||f_full - f_pruned||_inf`` is then ``max(|ub_f - lb_p|, |lb_f - ub_p|)``
unioned across cases.

This is the abcrown-style ``Zono-Sub`` analogue: two independent CROWN passes
subtracted at the end. There is no shared-zonotope cancellation, so the
bound is expected to be looser than ``zono3`` in :mod:`veri_pruning_diff`.

Usage::

    python abcrown_pruning_diff.py
    python abcrown_pruning_diff.py --depth 1 --eps 0.001 --K 14 --n-samples 3
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Locate ../abcrown — same lookup logic as the old certify_with_abcrown.
_DEFAULT_ABCROWN = (_HERE.parent.parent.parent / "abcrown").resolve()
_ABCROWN_DIR = Path(os.environ.get("ABCROWN_DIR", _DEFAULT_ABCROWN))
if not (_ABCROWN_DIR / "complete_verifier" / "abcrown.py").is_file():
    raise FileNotFoundError(
        f"abcrown not found at {_ABCROWN_DIR}. Set ABCROWN_DIR or place the repo at ../abcrown."
    )
for sub in (_ABCROWN_DIR, _ABCROWN_DIR / "complete_verifier"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from mnist_vit import mnist_vit as mnist_vit_clean
from veri_pruning_diff import (
    build_token_mask, classify_topk, enumerate_cases, monte_carlo, _masked_forward,
)
from vit_pruning_diff import scoring_model
import boundlab.expr as _expr  # noqa: F401
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.expr._affine import ConstVal
from boundlab.expr._var import LpEpsilon


# ---------------------------------------------------------------------------
# Concrete model wrappers (no boundlab custom ops — only stock torch).
# ---------------------------------------------------------------------------

class MaskedViT(nn.Module):
    """Batched ViT with concrete y-branch attention masking, rewritten so
    auto_LiRPA / CROWN can trace it cleanly.

    Differences from the reference ``mnist_vit.ViT``:

    * Leading batch dim throughout — every tensor has shape ``(B, ...)``.
    * Multi-head reshape is ``(B, N, H, D) → permute(0, 2, 1, 3) → (B, H, N, D)``
      instead of the no-batch ``(N, H, D) → (H, N, D)`` form. This is the
      change that lets CROWN's backward shape arithmetic survive.
    * Attention softmax is written explicitly as ``exp / (keep * exp).sum()``
      with no max-shift, so the trace has no ``ReduceMax`` node.

    Mask semantics match ``boundlab.diff.op.softmax_pruning``'s y-branch:

        attn_w[b, h, q, k] = exp(dots[b, h, q, k])
                           / sum_{k'} keep[k'] * exp(dots[b, h, q, k'])

    Pass ``token_mask=None`` for the unpruned baseline (all-keep mask =
    standard softmax).
    """

    def __init__(self, vit, token_mask: Tensor | None = None):
        super().__init__()
        self.vit = vit
        # keep_k broadcast shape: (1, 1, 1, N+1) — applies to (B, H, Q, K).
        if token_mask is None:
            keep_k = torch.ones(1, 1, 1, 17)
        else:
            keep_k = (token_mask > 0).float().view(1, 1, 1, -1)
        self.register_buffer("keep_k", keep_k)

    def forward(self, img: Tensor) -> Tensor:  # (B, C, H, W) → (B, num_classes)
        vit = self.vit
        B, C, H, W = img.shape
        patchify = vit.to_patch_embedding[0]
        embed = vit.to_patch_embedding[1]
        p = patchify.patch_size
        hh, ww = H // p, W // p

        # Batched patchify: (B, C, hh*p, ww*p) → (B, hh*ww, p*p*C).
        x = img.reshape(B, C, hh, p, ww, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(B, hh * ww, p * p * C)
        x = embed(x)                                       # (B, N, D)

        # Prepend CLS, add positional embedding.
        cls = vit.cls_token.expand(B, -1, -1)              # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                     # (B, N+1, D)
        x = x + vit.pos_embedding                          # (1, N+1, D) broadcasts

        for attn_block, ff_block in vit.transformer.layers:
            x = self._attn(x, attn_block)
            x = self._ff(x, ff_block)

        x = x.mean(dim=1) if vit.pool == "mean" else x[:, 0]
        return vit.mlp_head(x)

    def _attn(self, x: Tensor, attn_block) -> Tensor:
        """Batched attention with y-branch denominator masking."""
        prenorm = attn_block.fn
        attn = prenorm.fn
        residual = x
        xn = prenorm.norm(x)                               # (B, N, D)
        B, N, _ = xn.shape
        h, d = attn.heads, attn.dim_head

        q = attn.to_q(xn).reshape(B, N, h, d).permute(0, 2, 1, 3)  # (B, H, N, D)
        k = attn.to_k(xn).reshape(B, N, h, d).permute(0, 2, 1, 3)
        v = attn.to_v(xn).reshape(B, N, h, d).permute(0, 2, 1, 3)

        dots = (q @ k.transpose(-2, -1)) * attn.scale      # (B, H, N, N)
        exp_dots = dots.exp()                              # no max-shift
        denom = (exp_dots * self.keep_k).sum(dim=-1, keepdim=True)
        attn_w = exp_dots / denom

        out = (attn_w @ v).permute(0, 2, 1, 3).reshape(B, N, h * d)
        out = attn.to_out(out)
        return residual + out

    def _ff(self, x: Tensor, ff_block) -> Tensor:
        prenorm = ff_block.fn
        return x + prenorm.fn(prenorm.norm(x))


def FullViT(vit):
    return MaskedViT(vit, token_mask=None)


def PrunedViT(vit, token_mask: Tensor):
    return MaskedViT(vit, token_mask=token_mask)


# ---------------------------------------------------------------------------
# auto_LiRPA wrapper
# ---------------------------------------------------------------------------

def _crown_bounds(module: nn.Module, img: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """Per-class (lb, ub) via CROWN, falling back to IBP if CROWN trips."""
    dummy = img.unsqueeze(0)                         # (1, C, H, W)
    bnd = BoundedModule(module, dummy, device="cpu", bound_opts={"conv_mode": "matrix"})
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    x = BoundedTensor(dummy, ptb)
    try:
        lb, ub = bnd.compute_bounds(x=(x,), method="CROWN")
    except (NotImplementedError, AssertionError, RuntimeError):
        lb, ub = bnd.compute_bounds(x=(x,), method="IBP")
    return lb.squeeze(0).detach(), ub.squeeze(0).detach()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_samples(n: int, data_dir: Path, seed: int):
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(str(data_dir), train=False, download=True,
                            transform=transforms.ToTensor())
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
        return [(ds[i][0], int(ds[i][1])) for i in idx]
    except Exception as e:
        print(f"[data] MNIST unavailable ({e}); using synthetic", flush=True)
        g = torch.Generator().manual_seed(seed)
        return [(torch.rand(1, 28, 28, generator=g), -1) for _ in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(depth: int = 1, eps: float = 0.004, K: int = 14,
        n_samples: int = 3, mc_samples: int = 200, seed: int = 0):
    torch.manual_seed(seed)
    num_tokens = 16

    print(f"[setup] depth={depth} K={K} eps={eps}", flush=True)
    t0 = time.time()
    vit = mnist_vit_clean(depth=depth).eval()
    full_module = FullViT(vit).eval()
    # Cache full-model bounds once per image (mask-independent).
    op_score = zono.interpret(str(scoring_model(depth=depth)))
    print(f"[setup] done in {time.time() - t0:.1f}s\n")

    samples = load_samples(n_samples, _HERE / "mnist_data", seed)
    full_mask = build_token_mask(num_tokens, set(range(num_tokens)))

    hdr = (f"{'#':>3}  {'lbl':>3} {'pred':>4}  "
           f"{'keep':>4}/{'prn':>3}/{'unc':>3}  {'cases':>5}  "
           f"{'MC':>10}  {'abcrown':>10}  {'t_MC':>6} {'t_v':>6}")
    print(hdr)
    print("-" * len(hdr))

    all_mc, all_ab = [], []

    for i, (img, label) in enumerate(samples):
        with torch.no_grad():
            pred = int(vit(img).argmax().item())

        # --- Score-based case enumeration (same as veri_pruning_diff). ---
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        img_zono = ConstVal(img) + eps * LpEpsilon(list(img.shape))
        score_expr = op_score(img_zono)
        ub_mul, lb_mul = score_expr.ublb()
        ub_scores = ub_mul[:, 0, 1:].mean(0)
        lb_scores = lb_mul[:, 0, 1:].mean(0)
        d_keep, d_prn, unc = classify_topk(ub_scores, lb_scores, K)
        cases = enumerate_cases(d_keep, unc, K)

        # --- Full-model bounds via CROWN (mask-independent). ---
        t = time.perf_counter()
        full_lb, full_ub = _crown_bounds(full_module, img, eps)
        t_v_total = time.perf_counter() - t

        # --- MC (concrete, y-branch semantics). ---
        t_mc_total = 0.0
        mc_best = 0.0
        ab_ub_best = ab_lb_best = None

        for kept in cases:
            mask = build_token_mask(num_tokens, kept)

            t = time.perf_counter()
            mc = monte_carlo(vit, img, eps, full_mask, mask, n_mc=mc_samples)
            t_mc_total += time.perf_counter() - t
            mc_best = max(mc_best, mc)

            pruned_module = PrunedViT(vit, mask).eval()
            t = time.perf_counter()
            prn_lb, prn_ub = _crown_bounds(pruned_module, img, eps)
            t_v_total += time.perf_counter() - t

            d_ub = full_ub - prn_lb
            d_lb = full_lb - prn_ub
            if ab_ub_best is None:
                ab_ub_best, ab_lb_best = d_ub.clone(), d_lb.clone()
            else:
                ab_ub_best = torch.maximum(ab_ub_best, d_ub)
                ab_lb_best = torch.minimum(ab_lb_best, d_lb)

        ab_bound = max(ab_ub_best.abs().max().item(), ab_lb_best.abs().max().item())
        all_mc.append(mc_best); all_ab.append(ab_bound)

        lbl = f"{label:>3}" if label >= 0 else "  -"
        print(f"{i+1:>3}  {lbl} {pred:>4}  "
              f"{len(d_keep):>4}/{len(d_prn):>3}/{len(unc):>3}  "
              f"{len(cases):>5}  "
              f"{mc_best:>10.6f}  {ab_bound:>10.4f}  "
              f"{t_mc_total:>6.1f} {t_v_total:>6.1f}")

    # --- Summary ---
    n = len(samples)
    print()
    print("=" * 72)
    print(f"  abcrown / auto_LiRPA Pruning Verification — MNIST ViT (depth={depth})")
    print(f"  eps={eps}, K={K}/{num_tokens}, MC={mc_samples}, n={n}")
    print("=" * 72)
    print(f"  {'MC (empirical)':<20} {sum(all_mc) / n:>10.6f}  (lower bound)")
    print(f"  {'abcrown (CROWN-Sub)':<20} {sum(all_ab) / n:>10.4f}  (sound upper bound)")

    print(f"\n  Soundness: MC <= abcrown bound")
    all_sound = True
    for i in range(n):
        ok = all_mc[i] <= all_ab[i] + 1e-6
        if not ok:
            all_sound = False
        tag = "OK" if ok else "FAIL"
        print(f"    sample {i+1}: MC={all_mc[i]:.6f}  abcrown={all_ab[i]:.4f}  [{tag}]")
    print(f"\n  {'All ' + str(n) + ' samples sound.' if all_sound else 'WARNING: soundness violation!'}\n")


def main():
    ap = argparse.ArgumentParser(
        description="auto_LiRPA / abcrown baseline for differential pruning verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--depth", type=int, default=1, choices=[1, 3])
    ap.add_argument("--eps", type=float, default=0.004)
    ap.add_argument("--K", type=int, default=14)
    ap.add_argument("--n-samples", type=int, default=3, dest="n_samples")
    ap.add_argument("--mc-samples", type=int, default=200, dest="mc_samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
