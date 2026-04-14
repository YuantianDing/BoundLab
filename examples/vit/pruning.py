"""Differential verification helpers for the pruning-aware ViT.

This module shows how to bound the output difference between an unpruned
Vision Transformer and its pruning-aware variant defined in
``examples/vit/vit_threshold.py`` using the differential zonotope
interpreter.  It focuses on the ViT-specific plumbing; unit tests live in
``tests/test_vit_pruning_example.py``.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple

import torch

import boundlab.expr as expr
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export

from . import vit_threshold

ViTFactory = Callable[..., torch.nn.Module]

__all__ = [
    "diff_verify_pruned_vit",
]


def _make_vit_pair(
    ctor: ViTFactory,
    layer_norm_type: Literal["standard", "no_var"],
    pruning_threshold: Optional[float],
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Instantiate (baseline, pruned) ViT models with shared weights."""
    base = ctor(layer_norm_type=layer_norm_type, pruning_threshold=None).eval()
    pruned = ctor(
        layer_norm_type=layer_norm_type,
        pruning_threshold=pruning_threshold,
    ).eval()
    # Share weights so only the pruning logic differs.
    pruned.load_state_dict(base.state_dict())
    return base, pruned


def diff_verify_pruned_vit(
    *,
    ctor: ViTFactory = vit_threshold.vit_ibp_3_3_8,
    layer_norm_type: Literal["standard", "no_var"] = "no_var",
    pruning_threshold: float = 0.05,
    eps: float = 0.002,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run differential verification between unpruned and pruned ViT.

    Returns (diff_ub, diff_lb) tensors bounding f(x) - g(x), where f is the
    baseline model and g is its pruning-aware counterpart with identical
    weights but pruning threshold ``pruning_threshold``.
    """
    torch.manual_seed(seed)
    base, pruned = _make_vit_pair(ctor, layer_norm_type, pruning_threshold)

    # Export both models (shared input shape: C=3, H=W=32).
    onnx_base = onnx_export(base, ([3, 32, 32],))
    onnx_pruned = onnx_export(pruned, ([3, 32, 32],))
    merged = diff_net(onnx_base, onnx_pruned)
    op = diff_interpret(merged)

    # Shared input noise (tighter bounds than independent noise symbols).
    center = torch.randn(3, 32, 32) * 0.05
    eps_expr = expr.LpEpsilon(list(center.shape))
    x = expr.ConstVal(center) + eps * eps_expr
    triple = DiffExpr3(x, x, expr.ConstVal(torch.zeros_like(center)))

    out = op(triple)
    diff_ub, diff_lb = out.diff.ublb()
    return diff_ub, diff_lb


if __name__ == "__main__":
    ub, lb = diff_verify_pruned_vit()
    width = (ub - lb).max().item()
    print(f"Max output diff width: {width:.4f}")
