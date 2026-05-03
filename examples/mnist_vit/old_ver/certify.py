"""Certify the MNIST ViT's L∞ robustness using BoundLab zonotope verification.

Uses a **split-pipeline** strategy to avoid the ``Cat`` expression node that
arises from ``torch.cat(cls_token, patches)``.  Instead of exporting the full
ViT as a single ONNX graph, we:

1. Export the **patchify** stage (pixel → patch embeddings) and propagate the
   pixel-space zonotope through it.
2. **Manually** prepend the cls token via ``PadOp + Add`` — this produces a
   clean ``AffineSum`` with ``LpEpsilon`` children (symmetric, no ``Cat``),
   so ``symmetric_decompose`` in the bilinear matmul handler works correctly.
3. Export the **post-concat** stage (pos embedding → transformer → head) and
   propagate the manually-constructed zonotope through it.

This gives **tight** bounds (no looseness from the ``ublb()``-based fix) while
remaining fully **sound** (no ``Cat``-induced center error).

Usage::

    python certify.py --checkpoint mnist_transformer.pt --eps 0.005 --n-samples 20
    python certify.py --no-normalize --eps 0.01
"""
from __future__ import annotations

import argparse
import sys
import time
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
# Split-pipeline sub-models
# ---------------------------------------------------------------------------

class PatchifyStage(nn.Module):
    """(C, H, W) → (num_patches, dim).  Pure linear, no cls token."""

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


class PostConcatStage(nn.Module):
    """(num_patches+1, dim) → (num_classes,).  Transformer + head."""

    def __init__(self, vit):
        super().__init__()
        self.transformer = vit.transformer
        self.pool = vit.pool
        self.mlp_head = vit.mlp_head

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.transformer.layers:
            x = attn(x)
            x = ff(x)
        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)


# ---------------------------------------------------------------------------
# Core: build zonotope without Cat
# ---------------------------------------------------------------------------

def build_zonotope_no_cat(vit, img, eps, op_patch, normalize, mean, std):
    """Propagate pixel-space L∞ ball through patchify, then manually prepend
    cls token via Pad+Add (no Cat node)."""
    num_patches = (img.shape[-1] // vit.patch_size) ** 2
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    # Stage 1: patchify
    patch_zono = op_patch(
        expr.ConstVal(img) + eps * expr.LpEpsilon(list(img.shape))
    )

    # Stage 2: Pad patches (N, D) → (N+1, D), add cls + pos
    pad_op = PadOp(patch_zono.shape, [0, 0, 1, 0])
    padded = AffineSum((pad_op, patch_zono))
    cls_padded = F.pad(vit.cls_token[0], [0, 0, 0, num_patches])
    return padded + ConstVal(cls_padded + vit.pos_embedding[0])


# ---------------------------------------------------------------------------
# Data loading
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
# Certification
# ---------------------------------------------------------------------------

def certify(
    checkpoint, eps, n_samples, seed, data_dir,
    normalize, mean, std,
):
    torch.manual_seed(seed)
    vit = build_mnist_vit(checkpoint)
    num_patches = (28 // vit.patch_size) ** 2
    dim = vit.pos_embedding.shape[-1]

    # Build interpreters once, reuse per sample.
    print("[export] building split-pipeline interpreters ...", flush=True)
    t0 = time.time()
    patchify = PatchifyStage(vit, normalize, mean, std).eval()
    gm_patch = onnx_export(patchify, ([1, 28, 28],))
    op_patch = zono.interpret(gm_patch)

    post = PostConcatStage(vit).eval()
    gm_post = onnx_export(post, ([num_patches + 1, dim],))
    op_post = zono.interpret(gm_post)
    print(f"[export] done in {time.time() - t0:.1f}s\n")

    # Concrete model for predictions.
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
    hdr = f"{'#':>3} {'label':>5} {'pred':>4} {'correct':>7} {'margin':>10}  {'time':>6}  status"
    print(hdr)
    print("-" * len(hdr))

    for i, (img, label) in enumerate(samples):
        with torch.no_grad():
            pred = int(concrete(img).argmax().item())
        correct = (label >= 0) and (pred == label)
        n_correct += int(correct)

        t0 = time.time()
        full_zono = build_zonotope_no_cat(
            vit, img, eps, op_patch, normalize, mean, std
        )
        ub, lb = op_post(full_zono).ublb()
        dt = time.time() - t0

        ub_others = ub.clone()
        ub_others[pred] = float("-inf")
        margin = float(lb[pred] - ub_others.max())
        certified = margin > 0.0
        n_certified += int(certified)

        tag = "CERT" if certified else "fail"
        lbl = f"{label:>5}" if label >= 0 else "    -"
        cor = "yes" if correct else ("  -" if label < 0 else " no")
        print(f"{i+1:>3} {lbl} {pred:>4} {cor:>7} {margin:>+10.4f} {dt:>5.1f}s  {tag}")

    print()
    print("=" * 58)
    print(f"  checkpoint     : {checkpoint}")
    print(f"  eps (L∞ pixel) : {eps}")
    norm_str = f"on (μ={mean}, σ={std})" if normalize else "off"
    print(f"  normalization  : {norm_str}")
    print(f"  samples        : {n_samples}")
    if any(l >= 0 for _, l in samples):
        print(f"  clean accuracy : {n_correct}/{n_samples}"
              f" = {100*n_correct/n_samples:.1f}%")
    print(f"  certified      : {n_certified}/{n_samples}"
          f" = {100*n_certified/n_samples:.1f}%")
    print("=" * 58)
    return {"n_total": n_samples, "n_correct": n_correct,
            "n_certified": n_certified}


def main():
    ap = argparse.ArgumentParser(
        description="BoundLab zonotope certification for MNIST ViT "
                    "(split pipeline, no Cat node).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", default="mnist_transformer.pt")
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--n-samples", type=int, default=20, dest="n_samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="./mnist_data", dest="data_dir")
    ap.add_argument("--no-normalize", dest="normalize",
                    action="store_false", default=True)
    ap.add_argument("--mean", type=float, default=0.1307)
    ap.add_argument("--std", type=float, default=0.3081)
    args = ap.parse_args()

    print("=" * 58)
    print("  BoundLab Certification — MNIST ViT (split pipeline)")
    print("=" * 58)
    print()
    certify(**vars(args))


if __name__ == "__main__":
    main()
