"""ViT adapter for BoundLab certification.

Loads the DeepT MNIST checkpoint (``mnist_transformer.pt``) with its original
``state_dict`` keys intact, but:

* drops the batch dimension — forward takes ``(C, H, W)`` and returns
  ``(num_classes,)``.  This matches the style of BoundLab's own
  ``examples/vit/vit.py``;
* replaces ``einops.Rearrange`` / ``rearrange`` / ``repeat`` with plain
  ``reshape`` and ``permute`` so ``torch.onnx.export`` traces cleanly into
  primitive ONNX ops that the zonotope interpreter dispatches on (``MatMul``,
  ``Softmax``, ``Relu``, ``ReduceMean``, ``Gather``, ``Concat``, ...);
* replaces ``nn.Dropout`` with ``nn.Identity`` (dropout is a no-op at eval
  time anyway, but Identity keeps the Sequential indices so state-dict keys
  line up with the original).

The module tree is intentionally identical to the DeepT ``vit.py`` the user
uploaded, so ``load_state_dict`` with the DeepT checkpoint succeeds
*strictly* (no missing / unexpected keys).
"""
from __future__ import annotations
import os

import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Layer-norm variants (bit-for-bit copies of the DeepT ones)
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
# Residual / PreNorm wrappers — same shapes as DeepT so keys align
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
# FeedForward — identical key layout (net.0 / net.3 hold the Linear weights,
# dropouts replaced by Identity so indices are preserved).
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
# Attention — plain reshape/permute in place of einops rearrange.
# No batch dim: input/output are (N, D).
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
        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),  # .0
                nn.Identity(),              # .1  (was Dropout)
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, D)
        n = x.shape[0]
        h, d = self.heads, self.dim_head

        q = self.to_q(x).reshape(n, h, d).permute(1, 0, 2)   # (h, N, d)
        k = self.to_k(x).reshape(n, h, d).permute(1, 0, 2)
        v = self.to_v(x).reshape(n, h, d).permute(1, 0, 2)

        dots = (q @ k.transpose(-2, -1)) * self.scale        # (h, N, N)
        attn = dots.softmax(dim=-1)
        out = attn @ v                                       # (h, N, d)
        out = out.permute(1, 0, 2).reshape(n, h * d)         # (N, h*d)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Transformer stack — same nesting as DeepT: layers[i][0]=attn-block,
# layers[i][1]=ff-block.
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, layer_norm_type: str):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head),
                                 layer_norm_type)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim),
                                 layer_norm_type)),
            ]))

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# ---------------------------------------------------------------------------
# Patchify — replaces einops.Rearrange for the patch embedding.
# No params, so state_dict is unaffected; lives at to_patch_embedding.0
# so that to_patch_embedding.1 remains the Linear (matching checkpoint keys).
# ---------------------------------------------------------------------------

class Patchify(nn.Module):
    """``(C, H, W)`` → ``(num_patches, C*p*p)`` via reshape + permute.

    Equivalent to einops::

        Rearrange('c (h p1) (w p2) -> (h w) (p1 p2 c)',
                  p1=patch_size, p2=patch_size)
    """

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, img: Tensor) -> Tensor:
        # img: (C, H, W)
        C, H, W = img.shape
        p = self.patch_size
        hh, ww = H // p, W // p
        x = img.reshape(C, hh, p, ww, p)              # (C, hh, p, ww, p)
        x = x.permute(1, 3, 2, 4, 0).contiguous()     # (hh, ww, p, p, C)
        return x.reshape(hh * ww, p * p * C)          # (N, patch_dim)


# ---------------------------------------------------------------------------
# ViT — no batch dim, cls-token pooling by default.
# Key layout matches the DeepT vit.py uploaded by the user exactly.
# ---------------------------------------------------------------------------

class ViT(nn.Module):
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
        layer_norm_type: str,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        assert pool in ("cls", "mean")

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Patchify(patch_size),                   # .0 — no params
            nn.Linear(patch_dim, dim),              # .1 — checkpoint key
        )
        self.patch_size = patch_size

        # Exactly the same parameter shapes as the DeepT checkpoint.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head,
                                       mlp_dim, layer_norm_type)

        self.pool = pool
        self.layer_norm_type = layer_norm_type

        self.mlp_head = nn.Sequential(
            _make_norm(dim, layer_norm_type),       # .0
            nn.Linear(dim, num_classes),            # .1
        )

    def forward(self, img: Tensor) -> Tensor:
        # img: (C, H, W)
        x = self.to_patch_embedding(img)                 # (N, D)
        cls = self.cls_token[0]                          # (1, D)
        x = torch.cat((cls, x), dim=0)                   # (N+1, D)
        x = x + self.pos_embedding[0]                    # (N+1, D)
        x = self.transformer(x)                          # (N+1, D)
        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)                          # (num_classes,)


# ---------------------------------------------------------------------------
# Factory for the DeepT MNIST checkpoint the user uploaded.
# ---------------------------------------------------------------------------

def build_mnist_vit(checkpoint_path: str | None = None) -> ViT:
    """Build the MNIST ViT matching ``mnist_transformer.pt``.

    Config matches the user's ``vit_certify.py``::

        image_size=28, patch_size=7, num_classes=10, channels=1,
        dim=64, depth=1, heads=4, mlp_dim=128, layer_norm_type="no_var"

    ``dim_head`` defaults to 64 in the DeepT code, giving
    ``inner_dim = 4 * 64 = 256`` — matches the 256×64 Q/K/V checkpoint shapes.
    """
    model = ViT(
        image_size=28,
        patch_size=7,
        num_classes=10,
        channels=1,
        dim=64,
        depth=1,
        heads=4,
        mlp_dim=128,
        layer_norm_type="no_var",
        pool="cls",
        dim_head=64,
    )
    if checkpoint_path is not None:
        DIRNAME = os.path.dirname(__file__)
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(DIRNAME, checkpoint_path)
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
    return model.eval()
