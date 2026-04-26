"""Differential verification helpers for the pruning-aware ViT.

This module shows how to bound the output difference between an unpruned
Vision Transformer and its pruning-aware variant defined in
``examples/vit/vit_threshold.py`` using the differential zonotope
interpreter.  It focuses on the ViT-specific plumbing; unit tests live in
``tests/test_vit_pruning_example.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import onnx_ir
import torch

import boundlab.expr as expr
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export

from . import vit_threshold
from . import vit


_X_ASSERT_RE = re.compile(r"^\(assert \((<=|>=) X_(\d+) ([^\s\)]+)\)\)\s*$")
_LABEL_RE = re.compile(r"label:\s*(\d+)")
_INPUT_SHAPE = (3, 32, 32)


def _parse_vnnlib_box(spec_path: Path, shape: Tuple[int, int, int] = _INPUT_SHAPE) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
    """Parse per-dimension input bounds from a VNNLIB file.

    Returns:
        center: midpoint tensor with ``shape``.
        radius: half-width tensor with ``shape``.
        label: parsed label from comment (if present).
    """
    flat_dim = shape[0] * shape[1] * shape[2]
    lb = torch.full((flat_dim,), float("-inf"), dtype=torch.float32)
    ub = torch.full((flat_dim,), float("inf"), dtype=torch.float32)
    label: Optional[int] = None

    with spec_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith(";") and label is None:
                m_label = _LABEL_RE.search(line)
                if m_label:
                    label = int(m_label.group(1))
            m = _X_ASSERT_RE.match(line)
            if m is None:
                continue
            op, idx_s, val_s = m.groups()
            idx = int(idx_s)
            if idx < 0 or idx >= flat_dim:
                continue
            val = float(val_s)
            if op == "<=":
                ub[idx] = val
            else:
                lb[idx] = val

    if not torch.isfinite(lb).all() or not torch.isfinite(ub).all():
        raise ValueError(f"Incomplete X bounds in VNNLIB: {spec_path}")
    if (ub < lb).any():
        raise ValueError(f"Inconsistent bounds (ub < lb) in VNNLIB: {spec_path}")

    center = ((lb + ub) * 0.5).view(*shape)
    radius = ((ub - lb) * 0.5).view(*shape)
    return center, radius, label


def _model_from_spec_name(spec_name: str) -> str:
    if spec_name.startswith("ibp_3_3_8"):
        return "ibp_3_3_8"
    if spec_name.startswith("pgd_2_3_16"):
        return "pgd_2_3_16"
    raise ValueError(f"Cannot infer model from spec filename: {spec_name}")


def diff_verify_pruned_vit(
    *,
    ctor: Callable[..., torch.nn.Module] = vit_threshold.vit_ibp_3_3_8,
    layer_norm_type: Literal["standard", "no_var"] = "no_var",
    pruning_threshold: float = 0.00,
    eps: float = 0.002,
    seed: int = 0,
    center: Optional[torch.Tensor] = None,
    radius: Optional[torch.Tensor] = None,
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

    # Use box bounds from VNNLIB when provided; otherwise random L∞ ball.
    if center is None:
        center = torch.randn(3, 32, 32) * 0.05
    eps_expr = expr.LpEpsilon(list(center.shape))
    if radius is None:
        x = expr.ConstVal(center) + eps * eps_expr
    else:
        if tuple(radius.shape) != tuple(center.shape):
            raise ValueError(
                f"center/radius shape mismatch: {tuple(center.shape)} vs {tuple(radius.shape)}"
            )
        x = expr.ConstVal(center) + radius * eps_expr

    out = op(x)
    diff_ub, diff_lb = out.diff.ublb()
    return diff_ub, diff_lb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differentially verify pruning-aware ViT vs unpruned baseline."
    )
    parser.add_argument(
        "--model",
        choices=["auto", "ibp_3_3_8", "pgd_2_3_16"],
        default="auto",
        help="Which ViT checkpoint to load (auto infers from spec filename).",
    )
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "vnnlib_seed1",
        help="Directory containing VNNLIB specs.",
    )
    parser.add_argument(
        "--specs",
        nargs="+",
        default=["ibp_3_3_8_167.vnnlib", "pgd_2_3_16_38.vnnlib"],
        help="VNNLIB specs to run (filenames under --spec-dir, or full paths).",
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

    ctor_map = {
        "ibp_3_3_8": vit.vit_ibp_3_3_8,
        "pgd_2_3_16": vit.vit_pgd_2_3_16,
    }

    for spec in args.specs:
        spec_path = Path(spec)
        if not spec_path.exists():
            spec_path = args.spec_dir / spec
        if not spec_path.exists():
            raise FileNotFoundError(f"Spec not found: {spec}")

        model_key = args.model if args.model != "auto" else _model_from_spec_name(spec_path.name)
        center, radius, label = _parse_vnnlib_box(spec_path)

        ub, lb = diff_verify_pruned_vit(
            ctor=ctor_map[model_key],
            layer_norm_type=args.layer_norm,
            pruning_threshold=args.pruning_threshold,
            eps=args.eps,
            seed=args.seed,
            center=center,
            radius=radius,
        )
        width = (ub - lb).max().item()

        print(f"Spec: {spec_path}")
        print(f"Model: {model_key}, layer_norm={args.layer_norm}, label={label}")
        print(f"Pruning threshold: {args.pruning_threshold}, seed: {args.seed}")
        print(f"Max output diff width: {width:.6f}")

        if not torch.isfinite(ub).all() or not torch.isfinite(lb).all():
            print("Warning: non-finite bounds encountered (NaN/Inf).")
