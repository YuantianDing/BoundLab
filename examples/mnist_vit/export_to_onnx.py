"""Export MNIST ViT sub-models to ONNX files.

Saves the patch embedding, scoring model, and pruned ViT (full + pruned
variants) as ``.onnx`` files for inspection or use with external tools.

Usage::

    python export_to_onnx.py
    python export_to_onnx.py --depth 3 --checkpoint mnist_transformer_3.pt
    python export_to_onnx.py --K 12 --out-dir ./onnx_models

The ONNX graphs contain ``boundlab::HeavisidePruning`` custom-domain ops.
BoundLab's interpreters handle these; external ONNX runtimes will not.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mnist_vit import ViT, build_mnist_vit
from pipeline import ScoringModel, PrunedViT
from token_pruning import build_token_scores, build_all_kept_scores


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


def _export_and_save(module, args, path: Path):
    """Export a module to ONNX and save to disk."""
    program = torch.onnx.export(module, args, verbose=False)
    program.save(str(path))
    size_kb = path.stat().st_size / 1024
    print(f"  {path.name:<40} {size_kb:>8.1f} KB")


def main():
    ap = argparse.ArgumentParser(
        description="Export MNIST ViT sub-models to ONNX files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--depth", type=int, default=1, choices=[1, 3])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--K", type=int, default=8,
                    help="Number of tokens to keep (for pruned model)")
    ap.add_argument("--out-dir", default="onnx_export", dest="out_dir",
                    help="Output directory for ONNX files")
    ap.add_argument("--score-layer", type=int, default=0, dest="score_layer")
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = ("mnist_transformer.pt" if args.depth == 1
                           else f"mnist_transformer_{args.depth}.pt")

    num_tokens, dim = 16, 64
    vit = _build_model(args.depth, args.checkpoint)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {args.depth}-layer MNIST ViT (K={args.K}) to {out_dir}/")
    print()

    # 1. Patch embedding: (1, 28, 28) → (16, 64)
    _export_and_save(
        vit.to_patch_embedding.eval(),
        (torch.rand(1, 28, 28),),
        out_dir / "patch_embedding.onnx",
    )

    # 2. Scoring model: (17, 64) → (16,)
    scoring = ScoringModel(vit, score_layer=args.score_layer).eval()
    _export_and_save(
        scoring,
        (torch.rand(num_tokens + 1, dim),),
        out_dir / "scoring_model.onnx",
    )

    # 3. PrunedViT (full — all tokens kept): (17, 64) → (10,)
    full_scores = build_all_kept_scores(num_tokens)
    full_model = PrunedViT(
        vit, full_scores,
        mask_from_layer=args.score_layer,
        for_verification=True,
    ).eval()
    _export_and_save(
        full_model,
        (torch.rand(num_tokens + 1, dim),),
        out_dir / "pruned_vit_full.onnx",
    )

    # 4. PrunedViT (pruned — keep first K tokens): (17, 64) → (10,)
    kept = set(range(args.K))
    pruned_scores = build_token_scores(num_tokens, kept)
    pruned_model = PrunedViT(
        vit, pruned_scores,
        mask_from_layer=args.score_layer,
        for_verification=True,
    ).eval()
    _export_and_save(
        pruned_model,
        (torch.rand(num_tokens + 1, dim),),
        out_dir / f"pruned_vit_k{args.K}.onnx",
    )

    print(f"\nDone. {len(list(out_dir.glob('*.onnx')))} files in {out_dir}/")
    print()
    print("Note: graphs contain boundlab::HeavisidePruning custom ops.")
    print("These are handled by BoundLab's interpreters but not by")
    print("external ONNX runtimes (e.g. onnxruntime).")


if __name__ == "__main__":
    main()