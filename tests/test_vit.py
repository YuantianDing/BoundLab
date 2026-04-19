"""ViT certification tests and differential verification.

Tests:
1. Conv2d patch embedding soundness
2. CLS token concatenation soundness
3. Full ViT (1 layer, no_var, random weights) end-to-end
4. ViT shape check
5. Differential verification: original vs weight-perturbed ViT
"""

import copy
import warnings

import torch
from torch import nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

warnings.filterwarnings("ignore")


class LayerNormNoVar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias


class Attention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=8):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
    def forward(self, x):
        n, _ = x.shape; h = self.heads
        q = self.to_q(x).reshape(n, h, -1).permute(1, 0, 2)
        k = self.to_k(x).reshape(n, h, -1).permute(1, 0, 2)
        v = self.to_v(x).reshape(n, h, -1).permute(1, 0, 2)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).permute(1, 0, 2).reshape(n, -1)
        return self.to_out(out)


class SmallViT(nn.Module):
    def __init__(self, image_size=32, patch_size=16, num_classes=10,
                 dim=16, depth=1, heads=2, mlp_dim=32, channels=3, dim_head=8):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_conv = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(dim))
        self.pos_embedding = nn.Parameter(torch.zeros(num_patches + 1, dim))
        self.attn_norm = LayerNormNoVar(dim)
        self.attn = Attention(dim, heads, dim_head)
        self.head_norm = LayerNormNoVar(dim)
        self.head_linear = nn.Linear(dim, num_classes)
    def forward(self, img):
        x = self.patch_conv(img.unsqueeze(0))
        x = x.flatten(2).squeeze(0).transpose(0, 1)
        cls = self.cls_token.unsqueeze(0)
        x = torch.cat([cls, x], dim=0)
        x = x + self.pos_embedding[:x.shape[0]]
        x = self.attn(self.attn_norm(x)) + x
        x = x.mean(dim=0)
        return self.head_linear(self.head_norm(x))


def _make_input(center, eps):
    return expr.ConstVal(center) + expr.LpEpsilon(list(center.shape)) * eps

def _sample_inputs(center, eps, n=2000):
    return center.unsqueeze(0) + eps * (torch.rand(n, *center.shape) * 2 - 1)

def _check_bounds(outputs, ub, lb, tol=1e-4):
    assert (ub >= lb - tol).all(), f"UB < LB: {(lb - ub).max():.6f}"
    assert (outputs <= ub.unsqueeze(0) + tol).all(), f"UB violated: {(outputs - ub.unsqueeze(0)).max():.6f}"
    assert (outputs >= lb.unsqueeze(0) - tol).all(), f"LB violated: {(lb.unsqueeze(0) - outputs).max():.6f}"


def test_vit_conv_patch_embed_sound():
    torch.manual_seed(700)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=16, stride=16)
        def forward(self, img):
            return self.conv(img.unsqueeze(0)).flatten(2).squeeze(0).transpose(0, 1)
    model = PatchEmbed().eval()
    center = torch.randn(3, 32, 32) * 0.1; eps = 0.01
    op = zono.interpret(onnx_export(model, ([3, 32, 32],)))
    ub, lb = op(_make_input(center, eps)).ublb()
    assert ub.shape == torch.Size([4, 16])
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in _sample_inputs(center, eps)])
    _check_bounds(outputs, ub, lb)


def test_vit_concat_cls_sound():
    torch.manual_seed(701)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    class CLSConcat(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.randn(16) * 0.1)
            self.pos = nn.Parameter(torch.randn(5, 16) * 0.1)
        def forward(self, x):
            return torch.cat([self.cls_token.unsqueeze(0), x], dim=0) + self.pos[:x.shape[0]+1]
    model = CLSConcat().eval()
    center = torch.randn(4, 16) * 0.1; eps = 0.02
    op = zono.interpret(onnx_export(model, ([4, 16],)))
    ub, lb = op(_make_input(center, eps)).ublb()
    assert ub.shape == torch.Size([5, 16])
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in _sample_inputs(center, eps)])
    _check_bounds(outputs, ub, lb)


def test_vit_full_small_sound():
    torch.manual_seed(702)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    model = SmallViT().eval()
    center = torch.randn(3, 32, 32) * 0.05; eps = 0.002
    op = zono.interpret(onnx_export(model, ([3, 32, 32],)))
    ub, lb = op(_make_input(center, eps)).ublb()
    assert ub.shape == torch.Size([10])
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in _sample_inputs(center, eps, 1000)])
    _check_bounds(outputs, ub, lb, tol=1e-3)


def test_vit_shape_check():
    torch.manual_seed(703)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    model = SmallViT().eval()
    center = torch.randn(3, 32, 32) * 0.05; eps = 0.001
    op = zono.interpret(onnx_export(model, ([3, 32, 32],)))
    ub, lb = op(_make_input(center, eps)).ublb()
    assert ub.shape == torch.Size([10])
    assert (ub >= lb).all()


def test_vit_differential_sound():
    torch.manual_seed(704)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    from boundlab.diff.net import diff_net
    from boundlab.diff.zono3.expr import DiffExpr3
    from boundlab.diff.zono3 import interpret as diff_interpret

    model1 = SmallViT().eval()
    model2 = copy.deepcopy(model1)
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * 0.005)
    model2.eval()
    center = torch.randn(3, 32, 32) * 0.05; eps = 0.002
    merged = diff_net(onnx_export(model1, ([3,32,32],)), onnx_export(model2, ([3,32,32],)))
    op = diff_interpret(merged)
    x_expr = _make_input(center, eps)
    out = op(DiffExpr3(x_expr, x_expr, expr.ConstVal(torch.zeros_like(center))))
    diff_ub, diff_lb = out.diff.ublb()
    assert diff_ub.shape == torch.Size([10])
    assert (diff_ub >= diff_lb).all()
    for _ in range(500):
        x = center + (torch.rand_like(center) * 2 - 1) * eps
        with torch.no_grad():
            d = model1(x) - model2(x)
        assert (d <= diff_ub + 1e-3).all(), f"UB violated: {(d - diff_ub).max():.6f}"
        assert (d >= diff_lb - 1e-3).all(), f"LB violated: {(diff_lb - d).max():.6f}"
