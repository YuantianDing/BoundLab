"""ViT model for VNN-COMP 2023 benchmarks.

No batch dimension: input (C, H, W) -> output (num_classes,).
Supports both 'standard' (full LayerNorm) and 'no_var' (mean-only) normalization.
"""

from typing import Literal
from pathlib import Path

import torch
from torch import nn, Tensor
from safetensors.torch import load_file


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class LayerNormNoVar(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(-1, keepdim=True)
        return self.weight * (x - u) + self.bias


def make_norm(dim: int, layer_norm_type: str) -> nn.Module:
    if layer_norm_type == "standard":
        return LayerNorm(dim)
    else:
        return LayerNormNoVar(dim)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 3, dim_head: int = 16):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, D)
        n, _ = x.shape
        h = self.heads
        q = self.to_q(x).reshape(n, h, -1).permute(1, 0, 2)  # (h, N, d)
        k = self.to_k(x).reshape(n, h, -1).permute(1, 0, 2)
        v = self.to_v(x).reshape(n, h, -1).permute(1, 0, 2)

        dots = (q @ k.transpose(-2, -1)) * self.scale  # (h, N, N)
        attn = dots.softmax(dim=-1)
        out = attn @ v  # (h, N, d)
        out = out.permute(1, 0, 2).reshape(n, -1)  # (N, h*d)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int,
                 layer_norm_type: str):
        super().__init__()
        self.attn_norm = make_norm(dim, layer_norm_type)
        self.attn = Attention(dim, heads, dim_head)
        self.ff_norm = make_norm(dim, layer_norm_type)
        self.ff = FeedForward(dim, mlp_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(self.attn_norm(x)) + x
        x = self.ff(self.ff_norm(x)) + x
        return x


class ViT(nn.Module):
    """Vision Transformer without batch dimension.

    Input: (C, H, W) image tensor
    Output: (num_classes,) logits
    """

    def __init__(self, *, image_size: int, patch_size: int, num_classes: int,
                 dim: int, depth: int, heads: int, mlp_dim: int,
                 layer_norm_type: Literal["standard", "no_var"],
                 channels: int = 3, dim_head: int = 16):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        self.patch_conv = nn.Conv2d(channels, dim, kernel_size=patch_size,
                                    stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(dim))
        self.pos_embedding = nn.Parameter(torch.zeros(num_patches + 1, dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, layer_norm_type)
            for _ in range(depth)
        ])

        self.head_norm = make_norm(dim, layer_norm_type)
        self.head_linear = nn.Linear(dim, num_classes)

    def forward(self, img: Tensor) -> Tensor:
        # img: (C, H, W)
        x = self.patch_conv(img.unsqueeze(0))       # (1, D, H', W')
        x = x.flatten(2).squeeze(0).transpose(0, 1) # (N, D)

        cls = self.cls_token.unsqueeze(0)            # (1, D)
        x = torch.cat([cls, x], dim=0)              # (N+1, D)
        x = x + self.pos_embedding[:x.shape[0]]

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=0)                           # (D,) mean pooling
        return self.head_linear(self.head_norm(x))   # (num_classes,)


_MODELS_DIR = Path(__file__).parent


def vit_ibp_3_3_8(layer_norm_type: Literal["standard", "no_var"] = "standard") -> ViT:
    model = ViT(image_size=32, patch_size=8, num_classes=10, channels=3,
                dim=48, depth=3, heads=3, mlp_dim=96, dim_head=16,
                layer_norm_type=layer_norm_type)
    sd = load_file(_MODELS_DIR / "ibp_3_3_8.safetensors")
    model.load_state_dict(sd)
    return model


def vit_pgd_2_3_16(layer_norm_type: Literal["standard", "no_var"] = "standard") -> ViT:
    model = ViT(image_size=32, patch_size=16, num_classes=10, channels=3,
                dim=48, depth=2, heads=3, mlp_dim=96, dim_head=16,
                layer_norm_type=layer_norm_type)
    sd = load_file(_MODELS_DIR / "pgd_2_3_16.safetensors")
    model.load_state_dict(sd)
    return model
