"""Poly-domain robustness certification for the CIFAR ViT examples.

Runs BoundLab's poly interpreter on models from ``examples/vit/vit.py`` and
reports whether each sampled input is certifiably robust to L-inf perturbations.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


def certify_vit_poly(
    *,
    model_name: str = "ibp_3_3_8",
    layer_norm_type: str = "no_var",
    eps: float = 0.002,
    n_samples: int = 5,
    seed: int = 0,
) -> dict:
    """Certify random CIFAR-shaped points with poly bound propagation."""
    torch.manual_seed(seed)

    ctor = {
        "ibp_3_3_8": vit_module.vit_ibp_3_3_8,
        "pgd_2_3_16": vit_module.vit_pgd_2_3_16,
    }[model_name]
    model = ctor(layer_norm_type=layer_norm_type).eval()

    print(f"Exporting {model_name} to ONNX ...", flush=True)
    t0 = time.time()
    gm = onnx_export(model, ([3, 32, 32],))
    op = poly.interpret(gm)
    print(f"  Export + compile done ({time.time() - t0:.1f}s)\\n")

    results = []
    for i in range(n_samples):
        center = torch.rand(3, 32, 32)

        with torch.no_grad():
            logits = model(center)
            predicted = int(logits.argmax().item())

        t1 = time.time()
        x = expr.ConstVal(center) + eps * expr.LpEpsilon([3, 32, 32])
        out = op(x)
        ub, lb = out.ublb()
        elapsed = time.time() - t1

        ub_others = ub.clone()
        ub_others[predicted] = float("-inf")
        margin = float(lb[predicted] - ub_others.max())
        certified = margin > 0.0

        results.append(
            {
                "sample": i + 1,
                "predicted": predicted,
                "certified": certified,
                "margin": margin,
                "elapsed": elapsed,
            }
        )

        tag = "CERTIFIED" if certified else "not certified"
        print(
            f"  [{i + 1}/{n_samples}] class={predicted:2d}  {tag}  "
            f"margin={margin:+.4f}  ({elapsed:.1f}s)"
        )

    n_certified = sum(r["certified"] for r in results)
    return {"n_certified": n_certified, "n_total": n_samples, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poly certification for ViT on CIFAR-10-shaped images."
    )
    parser.add_argument(
        "--model",
        choices=["ibp_3_3_8", "pgd_2_3_16"],
        default="ibp_3_3_8",
        help="ViT checkpoint to load (default: ibp_3_3_8).",
    )
    parser.add_argument(
        "--layer-norm",
        choices=["standard", "no_var"],
        default="no_var",
        dest="layer_norm",
        help="LayerNorm variant (default: no_var).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.002,
        help="L-inf perturbation radius (default: 0.002).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        dest="n_samples",
        help="Number of random test images (default: 5).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    args = parser.parse_args()

    print("=" * 58)
    print("  Poly Certification — Vision Transformer")
    print("=" * 58)
    print(f"  model       : {args.model}")
    print(f"  layer_norm  : {args.layer_norm}")
    print(f"  eps (Linf)  : {args.eps}")
    print(f"  n_samples   : {args.n_samples}")
    print(f"  seed        : {args.seed}")
    print("=" * 58)
    print()

    result = certify_vit_poly(
        model_name=args.model,
        layer_norm_type=args.layer_norm,
        eps=args.eps,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    n_c = result["n_certified"]
    n_t = result["n_total"]
    pct = 100 * n_c / n_t if n_t else 0.0
    print(f"\\nCertified {n_c}/{n_t} ({pct:.0f}%)")
