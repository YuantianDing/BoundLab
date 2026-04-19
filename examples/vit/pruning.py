"""Differential verification helpers for the pruning-aware ViT.

This module shows how to bound the output difference between an unpruned
Vision Transformer and its pruning-aware variant defined in
``examples/vit/vit_threshold.py`` using the differential zonotope
interpreter.  It focuses on the ViT-specific plumbing; unit tests live in
``tests/test_vit_pruning_example.py``.
"""

from __future__ import annotations

import argparse
from typing import Callable, Literal, Optional, Tuple

import onnx_ir
import torch

import boundlab.expr as expr
from boundlab.diff.zono3.expr import DiffExpr2, DiffExpr3
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.expr._core import Expr
from boundlab.interp.onnx import onnx_export

from . import vit_threshold
from . import vit

def diff_verify_pruned_vit(
    *,
    ctor: Callable[..., torch.nn.Module] = vit_threshold.vit_ibp_3_3_8,
    layer_norm_type: Literal["standard", "no_var"] = "no_var",
    pruning_threshold: float = 0.00,
    eps: float = 0.002,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run differential verification between unpruned and pruned ViT.

    A single exported model encodes both branches: ``heaviside_pruning`` keeps
    the **x** branch unpruned and applies the mask only to the **y** branch.
    Returns (diff_ub, diff_lb) bounding f(x) - g(x).
    """
    torch.manual_seed(seed)

    model = ctor(
        layer_norm_type=layer_norm_type,
        # pruning_threshold=pruning_threshold,
    ).eval()
    onnx_model = onnx_export(model, ([3, 32, 32],))
    onnx_ir.save(onnx_model, "vit_pruning.onnx")

    
    op = diff_interpret(onnx_model, verbose=True)

    # Shared input noise (tighter bounds than independent noise symbols).
    center = torch.randn(3, 32, 32) * 0.05
    eps_expr = expr.LpEpsilon(list(center.shape))
    x = expr.ConstVal(center) + eps * eps_expr

    out = op(x)
    diff_ub, diff_lb = out.diff.ublb()
    return diff_ub, diff_lb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differentially verify pruning-aware ViT vs unpruned baseline."
    )
    parser.add_argument(
        "--model",
        choices=["ibp_3_3_8", "pgd_2_3_16"],
        default="ibp_3_3_8",
        help="Which ViT checkpoint to load.",
    )
    parser.add_argument(
        "--layer-norm",
        choices=["standard", "no_var"],
        default="no_var",
        help="LayerNorm variant to use (matches checkpoint).",
    )
    parser.add_argument(
        "--pruning-threshold",
        type=float,
        default=0.05,
        help="Threshold applied to class attention scores (None disables pruning).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.002,
        help="L∞ radius on the input image tensor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for input center sampling.",
    )

    args = parser.parse_args()

    ctor = {
        "ibp_3_3_8": vit.vit_ibp_3_3_8,
        "pgd_2_3_16": vit.vit_pgd_2_3_16,
    }[args.model]

    ub, lb = diff_verify_pruned_vit(
        ctor=ctor,
        layer_norm_type=args.layer_norm,
        pruning_threshold=args.pruning_threshold,
        eps=args.eps,
        seed=args.seed,
    )
    width = (ub - lb).max().item()

    print(f"Model: {args.model}, layer_norm={args.layer_norm}")
    print(f"Pruning threshold: {args.pruning_threshold}, eps: {args.eps}, seed: {args.seed}")
    print(f"Max output diff width: {width:.6f}")

    if not torch.isfinite(ub).all() or not torch.isfinite(lb).all():
        print("Warning: non-finite bounds encountered (NaN/Inf).")
