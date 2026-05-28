"""MNIST test-set accuracy under top-K token pruning.

For each image, scores patches by CLS-row attention from the first transformer
layer's Q·K^T (matching ``vit_pruning_diff.scoring_model``), keeps the top-K
patches plus CLS, and runs the clean ViT with the y-branch ``softmax_pruning``
attention masking from :mod:`veri_pruning_diff`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mnist_vit import mnist_vit
from veri_pruning_diff import _masked_forward, build_token_mask


def _patch_scores(vit, img: torch.Tensor) -> torch.Tensor:
    """Per-patch importance: CLS row of layer-0 Q·K^T, averaged over heads.

    Returns ``[num_patches]`` (CLS column dropped).
    """
    x = vit.to_patch_embedding(img)
    cls = vit.cls_token[0]
    x = torch.cat((cls, x), dim=0) + vit.pos_embedding[0]
    attn_block = vit.transformer.layers[0][0]
    prenorm = attn_block.fn
    attn = prenorm.fn
    xn = prenorm.norm(x)
    n = xn.shape[0]
    h, d = attn.heads, attn.dim_head
    q = attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)
    k = attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
    dots = (q @ k.transpose(-2, -1)) * attn.scale     # [h, n, n]
    return dots.mean(0)[0, 1:]                        # [n-1] = [num_patches]


@torch.no_grad()
def evaluate(vit, ds, K: int, num_patches: int = 16) -> float:
    correct = 0
    for img, label in ds:
        scores = _patch_scores(vit, img)
        kept_patches = set(torch.topk(scores, K).indices.tolist())
        mask = build_token_mask(num_patches, kept_patches)
        if _masked_forward(vit, img, mask).argmax().item() == label:
            correct += 1
    return correct / len(ds)


def main() -> None:
    ds = datasets.MNIST(_HERE / "mnist_data", train=False, download=True,
                        transform=transforms.ToTensor())
    ks = (16, 14, 12, 10, 8, 6, 4, 2, 1)
    for depth in (1, 3):
        vit = mnist_vit(depth=depth).eval()
        print(f"mnist_vit depth={depth} ({len(ds)} samples):")
        for K in ks:
            acc = evaluate(vit, ds, K)
            print(f"  top-{K:>2}/16: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
