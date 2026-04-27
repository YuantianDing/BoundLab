"""Poly-domain robustness certification for the CIFAR ViT examples.

Runs BoundLab's poly interpreter on models from ``examples/vit/vit.py`` and
reports whether each VNNLib property is certifiably robust.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

# Allow importing local examples modules and project package when run as script.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import boundlab.expr as expr
import boundlab.poly as poly
from boundlab.interp.onnx import onnx_export
import vit as vit_module

_X_ASSERT_RE = re.compile(r"^\(assert \((<=|>=) X_(\d+) ([^\s\)]+)\)\)\s*$")
_Y_PAIR_RE = re.compile(r"\(>= Y_(\d+) Y_(\d+)\)")
_LABEL_RE = re.compile(r"label:\s*(\d+)")
_INPUT_SHAPE = (3, 32, 32)


def _parse_vnnlib_box(
    spec_path: Path, shape: Tuple[int, int, int] = _INPUT_SHAPE
) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
    """Parse per-dimension input bounds and optional target label from VNNLIB."""
    flat_dim = shape[0] * shape[1] * shape[2]
    lb = torch.full((flat_dim,), float("-inf"), dtype=torch.float32)
    ub = torch.full((flat_dim,), float("inf"), dtype=torch.float32)
    label: Optional[int] = None
    y_rhs: Optional[int] = None

    with spec_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith(";"):
                if label is None:
                    m_label = _LABEL_RE.search(line)
                    if m_label:
                        label = int(m_label.group(1))
                continue

            m = _X_ASSERT_RE.match(line)
            if m is not None:
                op, idx_s, val_s = m.groups()
                idx = int(idx_s)
                if 0 <= idx < flat_dim:
                    val = float(val_s)
                    if op == "<=":
                        ub[idx] = val
                    else:
                        lb[idx] = val
                continue

            for m_y in _Y_PAIR_RE.finditer(line):
                rhs = int(m_y.group(2))
                if y_rhs is None:
                    y_rhs = rhs
                elif y_rhs != rhs:
                    raise ValueError(f"Inconsistent output label in VNNLIB: {spec_path}")

    if not torch.isfinite(lb).all() or not torch.isfinite(ub).all():
        raise ValueError(f"Incomplete X bounds in VNNLIB: {spec_path}")
    if (ub < lb).any():
        raise ValueError(f"Inconsistent bounds (ub < lb) in VNNLIB: {spec_path}")
    if label is None:
        label = y_rhs
    elif y_rhs is not None and label != y_rhs:
        raise ValueError(f"Conflicting labels in VNNLIB comments/output: {spec_path}")

    center = ((lb + ub) * 0.5).view(*shape)
    radius = ((ub - lb) * 0.5).view(*shape)
    return center, radius, label


def _model_from_spec_name(spec_name: str) -> str:
    if spec_name.startswith("ibp_3_3_8"):
        return "ibp_3_3_8"
    if spec_name.startswith("pgd_2_3_16"):
        return "pgd_2_3_16"
    raise ValueError(f"Cannot infer model from spec filename: {spec_name}")


def _build_poly_op(model_name: str, layer_norm_type: str):
    ctor = {
        "ibp_3_3_8": vit_module.vit_ibp_3_3_8,
        "pgd_2_3_16": vit_module.vit_pgd_2_3_16,
    }[model_name]
    model = ctor(layer_norm_type=layer_norm_type).eval()
    print(f"Exporting {model_name} to ONNX ...", flush=True)
    t0 = time.time()
    gm = onnx_export(model, ([3, 32, 32],))
    op = poly.interpret(gm, verbose=True)
    print(f"  Export + compile done ({time.time() - t0:.1f}s)\n")
    return model, op


def certify_vit_poly_vnnlib(
    *,
    model_name: str = "auto",
    layer_norm_type: str = "no_var",
    spec_dir: Path = _HERE / "vnnlib_seed1",
    specs: Optional[list[str]] = None,
) -> dict:
    """Certify VNNLib robustness specs with poly bound propagation."""
    if specs is None:
        specs = ["ibp_3_3_8_167.vnnlib", "pgd_2_3_16_38.vnnlib"]

    results = []
    model_cache: dict[str, tuple[torch.nn.Module, object]] = {}
    resolved_specs = []
    for spec in specs:
        spec_path = Path(spec)
        if not spec_path.exists():
            spec_path = spec_dir / spec
        if not spec_path.exists():
            raise FileNotFoundError(f"Spec not found: {spec}")
        resolved_specs.append(spec_path)

    for i, spec_path in enumerate(resolved_specs):
        model_key = model_name if model_name != "auto" else _model_from_spec_name(spec_path.name)
        if model_key not in model_cache:
            model_cache[model_key] = _build_poly_op(model_key, layer_norm_type)
        model, op = model_cache[model_key]
        center, radius, label = _parse_vnnlib_box(spec_path)

        with torch.no_grad():
            logits = model(center)
            predicted = int(logits.argmax().item())

        t1 = time.time()
        x = expr.ConstVal(center) + radius * expr.LpEpsilon([3, 32, 32])
        out = op(x)
        ub, lb = out.ublb()
        elapsed = time.time() - t1

        target = predicted if label is None else label
        ub_others = ub.clone()
        ub_others[target] = float("-inf")
        margin = float(lb[target] - ub_others.max())
        certified = margin > 0.0

        results.append(
            {
                "sample": i + 1,
                "spec": str(spec_path),
                "target": target,
                "predicted": predicted,
                "certified": certified,
                "margin": margin,
                "elapsed": elapsed,
            }
        )

        tag = "CERTIFIED" if certified else "not certified"
        print(
            f"  [{i + 1}/{len(resolved_specs)}] {spec_path.name} "
            f"pred={predicted:2d} target={target:2d}  {tag}  "
            f"margin={margin:+.4f}  ({elapsed:.1f}s)"
        )

    n_certified = sum(r["certified"] for r in results)
    return {"n_certified": n_certified, "n_total": len(resolved_specs), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poly certification for ViT from VNNLIB specs."
    )
    parser.add_argument(
        "--model",
        choices=["auto", "ibp_3_3_8", "pgd_2_3_16"],
        default="auto",
        help="ViT checkpoint to load (auto infers from spec filename).",
    )
    parser.add_argument(
        "--layer-norm",
        choices=["standard", "no_var"],
        default="no_var",
        dest="layer_norm",
        help="LayerNorm variant (default: no_var).",
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
    args = parser.parse_args()

    print("=" * 58)
    print("  Poly Certification — Vision Transformer")
    print("=" * 58)
    print(f"  model       : {args.model}")
    print(f"  layer_norm  : {args.layer_norm}")
    print(f"  spec_dir    : {args.spec_dir}")
    print(f"  specs       : {len(args.specs)} file(s)")
    print("=" * 58)
    print()

    result = certify_vit_poly_vnnlib(
        model_name=args.model,
        layer_norm_type=args.layer_norm,
        spec_dir=args.spec_dir,
        specs=args.specs,
    )

    n_c = result["n_certified"]
    n_t = result["n_total"]
    pct = 100 * n_c / n_t if n_t else 0.0
    print(f"\\nCertified {n_c}/{n_t} ({pct:.0f}%)")
