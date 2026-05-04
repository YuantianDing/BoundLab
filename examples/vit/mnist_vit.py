"""MNIST ViT for BoundLab certification.

Architecture mirrors the DeepT MNIST checkpoint layout so ``load_state_dict``
succeeds strictly against weights saved in ``mnist_vit_1.safetensors`` (depth=1)
and ``mnist_vit_3.safetensors`` (depth=3).

No batch dimension: forward takes ``(C, H, W)`` and returns ``(num_classes,)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn, Tensor
from safetensors.torch import load_file


_MODELS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Layer-norm variants
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class LayerNormNoVar(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(-1, keepdim=True)
        return self.weight * (x - u) + self.bias


def _make_norm(dim: int, kind: str) -> nn.Module:
    assert kind in ("standard", "no_var")
    return LayerNorm(dim) if kind == "standard" else LayerNormNoVar(dim)


# ---------------------------------------------------------------------------
# Residual / PreNorm wrappers
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, layer_norm_type: str):
        super().__init__()
        self.norm = _make_norm(dim, layer_norm_type)
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))


# ---------------------------------------------------------------------------
# FeedForward — Identity stubs preserve Sequential indices from original ckpt.
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),   # .0
            nn.ReLU(),                    # .1
            nn.Identity(),                # .2  (was Dropout)
            nn.Linear(hidden_dim, dim),   # .3
            nn.Identity(),                # .4  (was Dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Attention — no bias on Q/K/V, plain reshape/permute (no einops).
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        project_out = not (heads == 1 and dim_head == dim)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Identity())  # .0 / .1
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        h, d = self.heads, self.dim_head

        q = self.to_q(x).reshape(n, h, d).permute(1, 0, 2)   # (h, N, d)
        k = self.to_k(x).reshape(n, h, d).permute(1, 0, 2)
        v = self.to_v(x).reshape(n, h, d).permute(1, 0, 2)

        dots = (q @ k.transpose(-2, -1)) * self.scale         # (h, N, N)
        attn = dots.softmax(dim=-1)
        out = (attn @ v).permute(1, 0, 2).reshape(n, h * d)   # (N, h*d)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Transformer stack — layers[i][0]=attn-block, layers[i][1]=ff-block.
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, layer_norm_type: str):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head),
                                 layer_norm_type)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim), layer_norm_type)),
            ])
            for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# ---------------------------------------------------------------------------
# Patchify — replaces einops.Rearrange, no parameters.
# Lives at to_patch_embedding.0; Linear is at to_patch_embedding.1.
# ---------------------------------------------------------------------------

class Patchify(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, img: Tensor) -> Tensor:
        C, H, W = img.shape
        p = self.patch_size
        hh, ww = H // p, W // p
        x = img.reshape(C, hh, p, ww, p)
        x = x.permute(1, 3, 2, 4, 0).contiguous()
        return x.reshape(hh * ww, p * p * C)


# ---------------------------------------------------------------------------
# ViT — no batch dim, cls-token or mean pooling.
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    """MNIST Vision Transformer.

    Input: ``(C, H, W)``  →  Output: ``(num_classes,)``
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        layer_norm_type: Literal["standard", "no_var"],
        pool: Literal["cls", "mean"] = "cls",
        channels: int = 1,
        dim_head: int = 64,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        assert pool in ("cls", "mean")

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Patchify(patch_size),          # .0
            nn.Linear(patch_dim, dim),     # .1
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       layer_norm_type)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            _make_norm(dim, layer_norm_type),  # .0
            nn.Linear(dim, num_classes),       # .1
        )

    def forward(self, img: Tensor) -> Tensor:
        x = self.to_patch_embedding(img)          # (N, D)
        cls = self.cls_token[0]                   # (1, D)
        x = torch.cat((cls, x), dim=0)            # (N+1, D)
        x = x + self.pos_embedding[0]
        x = self.transformer(x)
        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)


# ---------------------------------------------------------------------------
# Input normalisation wrapper
# ---------------------------------------------------------------------------

class _NormViT(nn.Module):
    def __init__(self, vit: ViT, mean: float, std: float):
        super().__init__()
        self.vit = vit
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x: Tensor) -> Tensor:
        return self.vit((x - self.mean) / self.std)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

DIRPATH = Path(__file__).parent


def _load_vit(cfg: dict, ckpt: Path) -> ViT:
    model = ViT(**cfg)
    model.load_state_dict(load_file(ckpt), strict=True)
    return model.eval()


def _resolve(checkpoint: str | Path | None, default: Path) -> Path:
    path = Path(default if checkpoint is None else checkpoint)
    return path if path.is_absolute() else DIRPATH / path


def mnist_vit(
    checkpoint: str | Path | None = None,
    layer_norm_type: Literal["standard", "no_var"] = "no_var",
    input_norm: tuple[float, float] | None = (0.1307, 0.3081),
) -> nn.Module:
    """1-layer MNIST ViT (depth=1, heads=4, dim=64).

    ``input_norm`` applies ``(x - mean) / std`` before the model; pass
    ``None`` to skip normalisation.
    """
    cfg = dict(image_size=28, patch_size=7, num_classes=10, channels=1,
               dim=64, depth=1, heads=4, mlp_dim=128,
               layer_norm_type=layer_norm_type, pool="cls", dim_head=64)
    model = _load_vit(cfg, _resolve(checkpoint, DIRPATH / "mnist_vit_1.safetensors"))
    if input_norm is not None:
        return _NormViT(model, *input_norm).eval()
    return model


def mnist_vit_3(
    checkpoint: str | Path | None = None,
    layer_norm_type: Literal["standard", "no_var"] = "no_var",
    input_norm: tuple[float, float] | None = (0.1307, 0.3081),
) -> nn.Module:
    """3-layer MNIST ViT (depth=3, heads=4, dim=64).

    ``input_norm`` applies ``(x - mean) / std`` before the model; pass
    ``None`` to skip normalisation.
    """
    cfg = dict(image_size=28, patch_size=7, num_classes=10, channels=1,
               dim=64, depth=3, heads=4, mlp_dim=128,
               layer_norm_type=layer_norm_type, pool="cls", dim_head=64)
    model = _load_vit(cfg, _resolve(checkpoint, DIRPATH / "mnist_vit_3.safetensors"))
    if input_norm is not None:
        return _NormViT(model, *input_norm).eval()
    return model
