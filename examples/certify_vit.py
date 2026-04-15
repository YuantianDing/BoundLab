"""DeepT-style ViT certification via zonotope bound propagation.

Certifies that the ViT model's top-1 prediction is robust to L∞ input
perturbations of radius *eps* around each test image.  The full forward
pass — Conv2d patch embedding, multi-head softmax attention (3 layers),
feed-forward blocks, and mean-pooling head — is bounded symbolically using
the DeepT softmax relaxation (Bonaert et al., NeurIPS 2021) together with
McCormick-style bilinear relaxation for the attention products.

A sample is *certified* when the symbolic lower bound on the predicted
class's logit exceeds the symbolic upper bound of every other class's logit
for ALL inputs in the L∞ ball of radius *eps*.

Reference:
    Bonaert et al., "Fast and Complete: Enabling Complete Neural Network
    Verification with Rapid and Massively Parallel Incomplete Verifiers",
    NeurIPS 2021.

Usage:
    python -m examples.certify_vit
    python -m examples.certify_vit --model pgd_2_3_16 --eps 0.001 --n-samples 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow `from vit.vit import ...` when run as a script or as a module.
_EXAMPLES_DIR = Path(__file__).parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import boundlab.expr as expr
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export
from vit import vit as vit_module  # examples/vit/vit.py


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def certify_vit(
    *,
    model_name: str = "ibp_3_3_8",
    layer_norm_type: str = "no_var",
    eps: float = 0.002,
    n_samples: int = 5,
    seed: int = 0,
) -> dict:
    """Run DeepT zonotope certification on random CIFAR-10-shaped inputs.

    Args:
        model_name:      ``"ibp_3_3_8"`` or ``"pgd_2_3_16"``.
        layer_norm_type: ``"no_var"`` (mean-only) or ``"standard"``.
        eps:             L∞ perturbation radius.
        n_samples:       Number of random images to certify.
        seed:            RNG seed for input generation.

    Returns:
        A dict with keys ``n_certified``, ``n_total``, and ``results``
        (a list of per-sample dicts).
    """
    torch.manual_seed(seed)

    ctor = {
        "ibp_3_3_8": vit_module.vit_ibp_3_3_8,
        "pgd_2_3_16": vit_module.vit_pgd_2_3_16,
    }[model_name]

    model = ctor(layer_norm_type=layer_norm_type).eval()

    # Export the ViT to ONNX IR once, then build the zonotope interpreter.
    # The interpreter can be reused across samples.
    print(f"Exporting {model_name} to ONNX ...", flush=True)
    t0 = time.time()
    gm = onnx_export(model, ([3, 32, 32],))
    op = zono.interpret(gm)
    print(f"  Export + compile done ({time.time() - t0:.1f}s)\n")

    results = []
    for i in range(n_samples):
        # Random image with pixel values in [0, 1].
        center = torch.rand(3, 32, 32)

        # Ground-truth (concrete) prediction.
        with torch.no_grad():
            logits = model(center)
            predicted = int(logits.argmax().item())

        # Build zonotope: center ± eps (L∞ ball).
        t1 = time.time()
        x = expr.ConstVal(center) + eps * expr.LpEpsilon([3, 32, 32])
        out = op(x)
        ub, lb = out.ublb()
        elapsed = time.time() - t1

        # Certified iff lb[predicted] > max_{j ≠ predicted} ub[j].
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
        print(f"  [{i + 1}/{n_samples}] class={predicted:2d}  {tag}  "
              f"margin={margin:+.4f}  ({elapsed:.1f}s)")

    n_certified = sum(r["certified"] for r in results)
    return {"n_certified": n_certified, "n_total": n_samples, "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepT zonotope certification for ViT on CIFAR-10-shaped images."
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
        help="L∞ perturbation radius (default: 0.002).",
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
    print("  DeepT Certification — Vision Transformer")
    print("=" * 58)
    print(f"  model       : {args.model}")
    print(f"  layer_norm  : {args.layer_norm}")
    print(f"  eps (L∞)    : {args.eps}")
    print(f"  n_samples   : {args.n_samples}")
    print(f"  seed        : {args.seed}")
    print("=" * 58)
    print()

    result = certify_vit(
        model_name=args.model,
        layer_norm_type=args.layer_norm,
        eps=args.eps,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    n_c = result["n_certified"]
    n_t = result["n_total"]
    pct = 100 * n_c / n_t if n_t else 0.0
    print(f"\nCertified {n_c}/{n_t} ({pct:.0f}%)")
