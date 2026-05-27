"""MNIST test-set accuracy for the 1-layer and 3-layer ViTs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mnist_vit import mnist_vit


@torch.no_grad()
def evaluate(net: torch.nn.Module, ds) -> float:
    correct = 0
    for img, label in ds:
        if net(img).argmax().item() == label:
            correct += 1
    return correct / len(ds)


def main() -> None:
    ds = datasets.MNIST(_HERE / "mnist_data", train=False, download=True,
                        transform=transforms.ToTensor())
    for depth in (1, 3):
        net = mnist_vit(depth=depth).eval()
        acc = evaluate(net, ds)
        print(f"mnist_vit depth={depth}: {acc * 100:.2f}% ({len(ds)} samples)")


if __name__ == "__main__":
    main()
