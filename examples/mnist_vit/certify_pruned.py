"""Certify token-pruning robustness for the MNIST ViT.

Given a top-K pruning strategy (keep K highest-importance patches based on CLS
attention weights, zero out the rest), certifies that the pruned model's output
is robust under L∞ pixel perturbations.

Uses zonotope certification with case splitting on uncertain tokens:
  1. Propagate zonotope to get bounds on CLS attention importance scores.
  2. Classify tokens as definite-keep / definite-prune / uncertain via pairwise
     bound comparisons.
  3. Case-split on uncertain tokens: enumerate C(U, K_remaining) combinations.
  4. For each case: mask pruned token embeddings to zero, add attention bias
     of -50 on pruned columns to remove them from softmax, certify.
  5. Union bounds across all cases.

Usage::

    python certify_pruned.py --checkpoint mnist_transformer.pt --eps 0.004 --K 8 --n-samples 20
    python certify_pruned.py --eps 0.01 --K 12 --n-samples 20
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.expr._affine import AffineSum, ConstVal
from boundlab.interp.onnx import onnx_export
from boundlab.linearop import PadOp

from mnist_vit import build_mnist_vit


# ---------------------------------------------------------------------------
# Split-pipeline sub-models (reused from certify.py)
# ---------------------------------------------------------------------------

class PatchifyStage(nn.Module):
    """(C, H, W) -> (num_patches, dim).  Pure linear, no cls token."""

    def __init__(self, vit, normalize=False, mean=0.0, std=1.0):
        super().__init__()
        self.patch_embed = vit.to_patch_embedding
        self.normalize = normalize
        if normalize:
            self.register_buffer("mean", torch.tensor(float(mean)))
            self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, img: Tensor) -> Tensor:
        if self.normalize:
            img = (img - self.mean) / self.std
        return self.patch_embed(img)


# ---------------------------------------------------------------------------
# Scoring model: extract CLS attention importance per patch
# ---------------------------------------------------------------------------

class ScoringModel(nn.Module):
    """(17, D) embeddings -> (16,) importance scores for patch tokens.

    Computes mean-over-heads CLS attention weights via softmax, returns the
    16 patch scores (excluding CLS-to-CLS self-attention).
    """

    def __init__(self, vit):
        super().__init__()
        attn_block = vit.transformer.layers[0][0]  # Residual(PreNorm(Attention))
        self.norm = attn_block.fn.norm
        self.attn = attn_block.fn.fn
        self.heads = self.attn.heads
        self.dim_head = self.attn.dim_head
        self.scale = self.attn.scale

    def forward(self, x: Tensor) -> Tensor:
        xn = self.norm(x)
        n = xn.shape[0]
        h, d = self.heads, self.dim_head

        Q = self.attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)  # (h, 17, d)
        K = self.attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)

        # CLS query only -> scores to all tokens
        Q_cls = Q[:, 0:1, :]                                # (h, 1, d)
        scores = (Q_cls @ K.transpose(-2, -1)) * self.scale  # (h, 1, 17)
        attn_weights = scores.softmax(dim=-1)                # (h, 1, 17)

        # Mean across heads, return patch scores (exclude CLS->CLS at idx 0)
        importance = attn_weights.mean(dim=0).squeeze(0)     # (17,)
        return importance[1:]                                # (16,)


# ---------------------------------------------------------------------------
# Masked post-concat model: transformer with attention bias for pruning
# ---------------------------------------------------------------------------

class MaskedPostConcat(nn.Module):
    """(17, D) -> (num_classes,) with attention bias to mask pruned tokens.

    The attn_bias is a (1, 1, 17) tensor:
      - 0.0 for kept tokens
      - -50.0 for pruned tokens
    Added to attention scores before softmax, making exp(-50) ~ 0 so pruned
    tokens vanish from the softmax denominator.
    """

    def __init__(self, vit, attn_bias: Tensor):
        super().__init__()
        attn_block = vit.transformer.layers[0][0]  # Residual(PreNorm(Attention))
        ff_block = vit.transformer.layers[0][1]    # Residual(PreNorm(FeedForward))

        # Attention components
        self.attn_norm = attn_block.fn.norm
        self.attn = attn_block.fn.fn
        self.heads = self.attn.heads
        self.dim_head = self.attn.dim_head
        self.scale = self.attn.scale

        # FF block (used as-is via Residual(PreNorm(FF)))
        self.ff_block = ff_block

        # Pool + head
        self.pool = vit.pool
        self.mlp_head = vit.mlp_head

        # Attention bias: constant, baked into ONNX
        self.register_buffer("attn_bias", attn_bias)  # (1, 1, 17)

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        h, d = self.heads, self.dim_head

        # --- Attention with bias ---
        residual = x
        xn = self.attn_norm(x)

        q = self.attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)  # (h, 17, d)
        k = self.attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
        v = self.attn.to_v(xn).reshape(n, h, d).permute(1, 0, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale  # (h, 17, 17)
        scores = scores + self.attn_bias                  # mask pruned columns
        attn_w = scores.softmax(dim=-1)                   # (h, 17, 17)
        out = (attn_w @ v).permute(1, 0, 2).reshape(n, h * d)
        out = self.attn.to_out(out)
        x = residual + out

        # --- FF block (unchanged) ---
        x = self.ff_block(x)

        # --- Pool + head ---
        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)


# ---------------------------------------------------------------------------
# Core: build zonotope without Cat (same as certify.py)
# ---------------------------------------------------------------------------

def build_zonotope_no_cat(vit, img, eps, op_patch):
    num_patches = (img.shape[-1] // vit.patch_size) ** 2
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    patch_zono = op_patch(
        expr.ConstVal(img) + eps * expr.LpEpsilon(list(img.shape))
    )

    pad_op = PadOp(patch_zono.shape, [0, 0, 1, 0])
    padded = AffineSum((pad_op, patch_zono))
    cls_padded = F.pad(vit.cls_token[0], [0, 0, 0, num_patches])
    return padded + ConstVal(cls_padded + vit.pos_embedding[0])


# ---------------------------------------------------------------------------
# Top-K classification via pairwise bound comparisons
# ---------------------------------------------------------------------------

def classify_topk(ub_scores, lb_scores, K):
    """Classify 16 patch tokens into definite-keep / definite-prune / uncertain.

    Token i is definite-keep if it provably beats at least (16 - K) others:
        lb_scores[i] > ub_scores[j] for at least (16 - K) values of j != i.
    Token i is definite-prune if at least K others provably beat it:
        ub_scores[i] < lb_scores[j] for at least K values of j != i.
    """
    N = len(ub_scores)  # 16
    n_prune = N - K     # how many get pruned

    # wins[i] = number of tokens that i provably beats
    # losses[i] = number of tokens that provably beat i
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

    definite_keep = set()
    definite_prune = set()

    for i in range(N):
        if wins[i] >= n_prune:
            # i provably beats enough tokens to guarantee top-K
            definite_keep.add(i)
        elif losses[i] >= K:
            # enough tokens provably beat i, so i is out of top-K
            definite_prune.add(i)

    uncertain = set(range(N)) - definite_keep - definite_prune
    return definite_keep, definite_prune, uncertain


# ---------------------------------------------------------------------------
# Case splitting and certification
# ---------------------------------------------------------------------------

def certify_pruned_sample(
    vit, img, eps, K, op_patch, op_score, op_post,
):
    """Certify one sample under top-K pruning.

    Returns: (ub, lb, n_cases, definite_keep, definite_prune, uncertain)
    """
    # Step 1: Build embedding zonotope
    full_zono = build_zonotope_no_cat(vit, img, eps, op_patch)

    # Step 2: Get importance score bounds
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()
    score_zono = op_score(full_zono)
    ub_scores, lb_scores = score_zono.ublb()

    # Step 3: Classify tokens (patch indices 0-15 map to embedding indices 1-16)
    definite_keep, definite_prune, uncertain = classify_topk(
        ub_scores, lb_scores, K
    )

    # How many remaining keep slots need to be filled from uncertain tokens
    K_remaining = K - len(definite_keep)

    if K_remaining < 0:
        # More definite-keeps than K — all are kept, none uncertain
        K_remaining = 0
        uncertain = set()

    if K_remaining > len(uncertain):
        # Not enough uncertain to fill — keep all uncertain
        K_remaining = len(uncertain)

    # Enumerate case splits: choose K_remaining from uncertain to keep
    uncertain_list = sorted(uncertain)

    if len(uncertain_list) == 0 or K_remaining == len(uncertain_list):
        # No real split: all uncertain are kept
        kept_patches = definite_keep | uncertain
        cases = [kept_patches]
    elif K_remaining == 0:
        # All uncertain are pruned
        cases = [definite_keep.copy()]
    else:
        cases = []
        for keep_combo in combinations(uncertain_list, K_remaining):
            kept_patches = definite_keep | set(keep_combo)
            cases.append(kept_patches)

    # Step 4: For each case, mask + certify
    best_lb = None
    best_ub = None

    for case_kept in cases:
        # Build mask: (17, 64) — CLS always 1, kept patches 1, pruned 0
        mask = torch.zeros(17, 64)
        mask[0] = 1.0  # CLS always kept
        for p in case_kept:
            mask[p + 1] = 1.0  # patch p -> embedding index p+1

        # Apply mask to embedding zonotope (exact: Hadamard with 0/1 constant)
        masked_zono = full_zono * mask

        # Propagate through standard PostConcat (reused across all cases)
        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()
        ub, lb = op_post(masked_zono).ublb()

        # Union across cases
        if best_lb is None:
            best_lb = lb.clone()
            best_ub = ub.clone()
        else:
            best_lb = torch.minimum(best_lb, lb)
            best_ub = torch.maximum(best_ub, ub)

    return best_ub, best_lb, len(cases), definite_keep, definite_prune, uncertain


# ---------------------------------------------------------------------------
# Data loading (same as certify.py)
# ---------------------------------------------------------------------------

def load_test_samples(n, data_dir, seed):
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(
            data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(ds), generator=g)[:n].tolist()
        samples = [(ds[i][0], int(ds[i][1])) for i in indices]
        print(f"[data] loaded {len(samples)} real MNIST test samples")
        return samples
    except Exception as e:
        print(f"[data] WARNING: MNIST unavailable ({type(e).__name__}: {e}).")
        print("[data] Falling back to synthetic inputs (smoke test only).")
        g = torch.Generator().manual_seed(seed)
        return [(torch.rand(1, 28, 28, generator=g), -1) for _ in range(n)]


# ---------------------------------------------------------------------------
# Main certification loop
# ---------------------------------------------------------------------------

def certify(
    checkpoint, eps, K, n_samples, seed, data_dir,
    normalize, mean, std,
):
    torch.manual_seed(seed)
    vit = build_mnist_vit(checkpoint)

    # Build interpreters
    print("[export] building interpreters ...", flush=True)
    t0 = time.time()

    patchify = PatchifyStage(vit, normalize, mean, std).eval()
    gm_patch = onnx_export(patchify, ([1, 28, 28],))
    op_patch = zono.interpret(gm_patch)

    scoring = ScoringModel(vit).eval()
    gm_score = onnx_export(scoring, ([17, 64],))
    op_score = zono.interpret(gm_score)

    from certify import PostConcatStage
    post = PostConcatStage(vit).eval()
    gm_post = onnx_export(post, ([17, 64],))
    op_post = zono.interpret(gm_post)

    print(f"[export] done in {time.time() - t0:.1f}s\n")

    # Concrete model for predictions
    if normalize:
        class Norm(nn.Module):
            def __init__(s):
                super().__init__()
                s.m = torch.tensor(float(mean))
                s.s = torch.tensor(float(std))
                s.vit = vit
            def forward(s, x): return s.vit((x - s.m) / s.s)
        concrete = Norm().eval()
    else:
        concrete = vit

    samples = load_test_samples(n_samples, data_dir, seed)
    print()

    n_correct = n_certified = 0
    total_cases = 0

    hdr = (f"{'#':>3} {'label':>5} {'pred':>4} {'correct':>7} "
           f"{'margin':>10}  {'cases':>5}  {'keep':>4}/{'prune':>5}/{'unc':>3}  "
           f"{'time':>5}  status")
    print(hdr)
    print("-" * len(hdr))

    for i, (img, label) in enumerate(samples):
        with torch.no_grad():
            pred = int(concrete(img).argmax().item())
        correct = (label >= 0) and (pred == label)
        n_correct += int(correct)

        t0 = time.time()
        ub, lb, n_cases, dk, dp, unc = certify_pruned_sample(
            vit, img, eps, K, op_patch, op_score, op_post,
        )
        dt = time.time() - t0

        ub_others = ub.clone()
        ub_others[pred] = float("-inf")
        margin = float(lb[pred] - ub_others.max())
        certified = margin > 0.0
        n_certified += int(certified)
        total_cases += n_cases

        tag = "CERT" if certified else "fail"
        lbl = f"{label:>5}" if label >= 0 else "    -"
        cor = "yes" if correct else ("  -" if label < 0 else " no")
        print(f"{i+1:>3} {lbl} {pred:>4} {cor:>7} {margin:>+10.4f}  "
              f"{n_cases:>5}  {len(dk):>4}/{len(dp):>5}/{len(unc):>3}  "
              f"{dt:>5.1f}s  {tag}")

    print()
    print("=" * 66)
    print(f"  checkpoint     : {checkpoint}")
    print(f"  eps (L-inf)    : {eps}")
    print(f"  K (keep)       : {K} of 16 patches")
    norm_str = f"on (mean={mean}, std={std})" if normalize else "off"
    print(f"  normalization  : {norm_str}")
    print(f"  samples        : {n_samples}")
    if any(l >= 0 for _, l in samples):
        print(f"  clean accuracy : {n_correct}/{n_samples}"
              f" = {100*n_correct/n_samples:.1f}%")
    print(f"  certified      : {n_certified}/{n_samples}"
          f" = {100*n_certified/n_samples:.1f}%")
    print(f"  total cases    : {total_cases}"
          f" (avg {total_cases/n_samples:.1f}/sample)")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser(
        description="BoundLab certification for top-K token pruning on MNIST ViT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", default="mnist_transformer.pt")
    ap.add_argument("--eps", type=float, default=0.004)
    ap.add_argument("--K", type=int, default=8,
                    help="Number of patch tokens to keep (out of 16)")
    ap.add_argument("--n-samples", type=int, default=20, dest="n_samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="./mnist_data", dest="data_dir")
    ap.add_argument("--no-normalize", dest="normalize",
                    action="store_false", default=True)
    ap.add_argument("--mean", type=float, default=0.1307)
    ap.add_argument("--std", type=float, default=0.3081)
    args = ap.parse_args()

    print("=" * 66)
    print(f"  BoundLab Certification -- MNIST ViT top-{args.K} token pruning")
    print("=" * 66)
    print()
    certify(**vars(args))


if __name__ == "__main__":
    main()