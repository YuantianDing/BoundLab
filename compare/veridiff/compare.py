"""Compare boundlab.diff.zono3 against VeriDiff on example networks.

Usage::

    python compare/veridiff/compare.py

Requires:
- onnx2torch  (pip install onnx2torch)
- Docker with image ``yuantianding/veridiff`` available

For each example the script computes:
- **BoundLab distance bound**: max over output dimensions of max(|ub_i|, |lb_i|)
  of the differential zonotope output using the input region from the spec.
- **VeriDiff Distance Bound**: the value printed by VeryDiff as
  ``Distance Bound: X.XX`` when run with ``--epsilon 0.1``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import onnx_ir
import torch

import boundlab.expr as expr
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as _diff_interpret
from boundlab.interp import Interpreter
from boundlab.diff.expr import DiffExpr2, DiffExpr3

# =====================================================================
# Set up interpreter — Softmax is skipped (treated as identity)
# =====================================================================
diff_interpret = Interpreter(_diff_interpret)
diff_interpret |= {
    "Softmax": lambda X, **_: X,
}

EXAMPLES_DIR = Path(__file__).parent / "examples"
NETS_DIR = EXAMPLES_DIR / "nets"
SPECS_DIR = EXAMPLES_DIR / "specs"


# =====================================================================
# vnnlib parser (input bounds only)
# =====================================================================

def parse_vnnlib_input_bounds(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(center, half_width)`` tensors for the input variables."""
    text = path.read_text()
    ubs: dict[int, float] = {}
    lbs: dict[int, float] = {}
    for m in re.finditer(r'\(assert \(<= X_(\d+) ([\d.eE+\-]+)\)\)', text):
        ubs[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r'\(assert \(>= X_(\d+) ([\d.eE+\-]+)\)\)', text):
        lbs[int(m.group(1))] = float(m.group(2))

    n = max(max(ubs, default=-1), max(lbs, default=-1)) + 1
    center = torch.tensor(
        [(lbs.get(i, 0.0) + ubs.get(i, 0.0)) / 2.0 for i in range(n)],
        dtype=torch.float32,
    )
    half_width = torch.tensor(
        [(ubs.get(i, 0.0) - lbs.get(i, 0.0)) / 2.0 for i in range(n)],
        dtype=torch.float32,
    )
    return center, half_width


# =====================================================================
# BoundLab computation
# =====================================================================

def _load_merged(net1_path: Path, net2_path: Path):
    """Merge the two ONNX networks and return a bound diff interpreter."""
    merged = diff_net(net1_path, net2_path)
    # onnx_ir.save(merged, "merged.onnx")
    return diff_interpret(merged)


def boundlab_distance_bound(
    net1_path: Path,
    net2_path: Path,
    center: torch.Tensor,
    half_width: torch.Tensor,
) -> float:
    """Compute the L∞ distance bound using differential zonotope propagation."""
    # Build zonotope for the 1-D input described by the spec
    eps = expr.LpEpsilon(list(center.shape))
    z = expr.ConstVal(center) + half_width * eps

    op = _load_merged(net1_path, net2_path)

    # Some batched models (e.g. with Softmax(dim=1)) need 2-D input
    try:
        out = op(z)
    except Exception:
        out = op(z.unsqueeze(0))

    # Extract the difference expression
    if isinstance(out, DiffExpr3):
        diff_expr = out.diff
    elif isinstance(out, DiffExpr2):
        diff_expr = out.x - out.y
    else:
        diff_expr = out

    ub = diff_expr.ub()
    lb = diff_expr.lb()
    return float(torch.max(torch.maximum(ub.abs(), lb.abs())).item())


# =====================================================================
# VeriDiff Docker runner
# =====================================================================

def veridiff_distance_bound(
    net1_path: Path,
    net2_path: Path,
    spec_path: Path,
    epsilon: float = 0.1,
    timeout: int = 300,
) -> float | None:
    """Run VeriDiff in Docker and extract the ``Distance Bound:`` value.

    Returns ``None`` if VeriDiff is unavailable or does not print a bound.
    """
    examples_abs = EXAMPLES_DIR.resolve()
    net1_rel = net1_path.resolve().relative_to(examples_abs)
    net2_rel = net2_path.resolve().relative_to(examples_abs)
    spec_rel = spec_path.resolve().relative_to(examples_abs)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{examples_abs}:/examples",
        "yuantianding/veridiff",
        f"/examples/{net1_rel}",
        f"/examples/{net2_rel}",
        f"/examples/{spec_rel}",
        "--epsilon", str(epsilon),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout + result.stderr
        m = re.search(r"Distance Bound:\s*([\d.eE+\-]+)", output)
        if m:
            return float(m.group(1))
        return None
    except FileNotFoundError:
        print("  [docker not found — skipping VeriDiff]", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  [VeriDiff timed out after {timeout}s]", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [VeriDiff error: {e}]", file=sys.stderr)
        return None


# =====================================================================
# Test cases
# =====================================================================

CASES = [
    {
        "name": "Single-layer 2x2 linear+ReLU (eps=0.1)",
        "net1": NETS_DIR / "single_layer_relu_ref.onnx",
        "net2": NETS_DIR / "single_layer_relu_alt.onnx",
        "spec": SPECS_DIR / "single_layer_box.vnnlib",
    },
    {
        "name": "2_80 (16-dim, σ=0.1)",
        "net1": NETS_DIR / "2_80-1.onnx",
        "net2": NETS_DIR / "2_80-1-0.1.onnx",
        "spec": SPECS_DIR / "sigma_0.1.vnnlib",
    },
    # {
    #     "name": "MNIST relu-3-100 vs pruned5 (img 0)",
    #     "net1": NETS_DIR / "mnist_relu_3_100.onnx",
    #     "net2": NETS_DIR / "mnist_relu_3_100_pruned5.onnx",
    #     "spec": SPECS_DIR / "mnist_0_local_15.vnnlib",
    # },
    # {
    #     "name": "MNIST relu-3-100 vs pruned5 (img 7)",
    #     "net1": NETS_DIR / "mnist_relu_3_100.onnx",
    #     "net2": NETS_DIR / "mnist_relu_3_100_pruned5.onnx",
    #     "spec": SPECS_DIR / "mnist_7_local_15.vnnlib",
    # },
    # {
    #     "name": "ACAS Xu 1_1 vs pruned5",
    #     "net1": NETS_DIR / "ACASXU_run2a_1_1_batch_2000.onnx",
    #     "net2": NETS_DIR / "ACASXU_run2a_1_1_batch_2000_pruned5.onnx",
    #     "spec": SPECS_DIR / "prop_1.vnnlib",
    # },
]


def main():
    col = 42
    print(f"{'Case':<{col}} {'BoundLab':>12} {'VeriDiff':>12} {'Ratio(BL/VD)':>14}")
    print("-" * (col + 42))

    for case in CASES:
        name = case["name"]
        net1, net2, spec = case["net1"], case["net2"], case["spec"]

        center, half_width = parse_vnnlib_input_bounds(spec)

        # BoundLab
        bl = boundlab_distance_bound(net1, net2, center, half_width)
        bl_str = f"{bl:.6f}"

        # VeriDiff
        vd = veridiff_distance_bound(net1, net2, spec)
        vd_str = f"{vd:.6f}" if vd is not None else "N/A"

        ratio_str = ""
        if bl is not None and vd is not None and vd > 0:
            ratio_str = f"{bl / vd:.3f}x"

        print(f"{name:<{col}} {bl_str:>12} {vd_str:>12} {ratio_str:>14}")


if __name__ == "__main__":
    main()
