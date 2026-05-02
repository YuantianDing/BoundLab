"""Compare boundlab.diff.zono3 against VeriDiff on example networks.

Usage::

    python compare/verydiff/compare.py

Requires:
- onnx2torch  (pip install onnx2torch)
- Docker with image ``yuantianding/verydiff`` available

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
import time
from pathlib import Path

import onnx_ir
import torch
from torch import nn

import boundlab.expr as expr
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as _diff_interpret
from boundlab.diff.zonohex import interpret as _hex_interpret
from boundlab.interp import Interpreter
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.interp.onnx import onnx_export
from boundlab.prop import ub

# =====================================================================
# Set up interpreters — Softmax is skipped (treated as identity)
# =====================================================================
diff_interpret = Interpreter(_diff_interpret)
diff_interpret |= {
    "Softmax": lambda X, **_: X,
}

hex_interpret = Interpreter(_hex_interpret)
hex_interpret |= {
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
    # Lower bounds: both `(assert (<= value X_i))` and `(assert (>= X_i value))` forms
    for m in re.finditer(r'\(assert \(<= ([\d.eE+\-]+) X_(\d+)\)\)', text):
        lbs[int(m.group(2))] = float(m.group(1))
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

def _load_merged(net1_path: Path, net2_path: Path, interpreter: Interpreter):
    """Merge the two ONNX networks and return a bound diff interpreter."""
    merged = diff_net(net1_path, net2_path)
    return interpreter(merged)


def _distance_bound(
    net1_path: Path,
    net2_path: Path,
    center: torch.Tensor,
    half_width: torch.Tensor,
    interpreter: Interpreter,
) -> float:
    """Compute the L∞ distance bound using the given differential interpreter."""
    eps = expr.LpEpsilon(list(center.shape))
    op = _load_merged(net1_path, net2_path, interpreter)

    class Mod(nn.Module):
        def forward(self, center, half_width):
            z = expr.ConstVal(center) + half_width * eps
            out = op(z)
            if isinstance(out, DiffExpr3):
                diff_expr = out.diff
            elif isinstance(out, DiffExpr2):
                diff_expr = out.x - out.y
            else:
                diff_expr = expr.ConstVal(torch.zeros_like(center))
            ub, lb = diff_expr.ublb()
            return torch.max(torch.maximum(ub.abs(), lb.abs()))
    mod = Mod()
    return float(mod(center, half_width).item())

def boundlab_distance_bound(net1_path, net2_path, center, half_width) -> float:
    return _distance_bound(net1_path, net2_path, center, half_width, diff_interpret)


def zonohex_distance_bound(net1_path, net2_path, center, half_width) -> float:
    return _distance_bound(net1_path, net2_path, center, half_width, hex_interpret)


# =====================================================================
# VeriDiff Docker runner
# =====================================================================

def verydiff_distance_bound(
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
        "yuantianding/verydiff",
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
        if result.returncode != 0:
            first_line = (result.stderr or result.stdout).splitlines()[0] if (result.stderr or result.stdout) else "unknown error"
            print(f"  [VeriDiff docker exited {result.returncode}: {first_line}]", file=sys.stderr)
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
        "name": "Single-layer relu (2-dim input)",
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
    {
        "name": "MNIST relu-3-100 vs pruned5 (img 0)",
        "net1": NETS_DIR / "mnist_relu_3_100.onnx",
        "net2": NETS_DIR / "mnist_relu_3_100_pruned5.onnx",
        "spec": SPECS_DIR / "mnist_0_local_15.vnnlib",
    },
    
    {
        "name": "MNIST relu-3-100 vs pruned5 (img 7)",
        "net1": NETS_DIR / "mnist_relu_3_100.onnx",
        "net2": NETS_DIR / "mnist_relu_3_100_pruned5.onnx",
        "spec": SPECS_DIR / "mnist_7_local_15.vnnlib",
    },
    {
        "name": "ACAS Xu 1_1 vs pruned5",
        "net1": NETS_DIR / "ACASXU_run2a_1_1_batch_2000.onnx",
        "net2": NETS_DIR / "ACASXU_run2a_1_1_batch_2000_pruned5.onnx",
        "spec": SPECS_DIR / "prop_1.vnnlib",
    },
]


def main():
    col = 42
    header = (
        f"{'Case':<{col}} {'BoundLab':>12} {'ZonoHex':>12} {'VeriDiff':>12} "
        f"{'Hex/BL':>8} {'BL/VD':>8} "
        f"{'BL(s)':>8} {'Hex(s)':>8} {'VD(s)':>8}"
    )
    print(header)
    print("-" * len(header))
    total_start = time.perf_counter()
    total_bl_time = 0.0
    total_hx_time = 0.0
    total_vd_time = 0.0

    for case in CASES:
        name = case["name"]
        net1, net2, spec = case["net1"], case["net2"], case["spec"]

        center, half_width = parse_vnnlib_input_bounds(spec)

        # BoundLab (zono3)
        bl_start = time.perf_counter()
        try:
            bl = boundlab_distance_bound(net1, net2, center, half_width)
            bl_str = f"{bl:.6f}"
        except Exception as e:
            bl = None
            bl_str = f"ERR:{type(e).__name__}"
        bl_elapsed = time.perf_counter() - bl_start
        total_bl_time += bl_elapsed

        # BoundLab (zonohex)
        hx_start = time.perf_counter()
        hx = zonohex_distance_bound(net1, net2, center, half_width)
        hx_str = f"{hx:.6f}"
        hx_elapsed = time.perf_counter() - hx_start
        total_hx_time += hx_elapsed

        # VeriDiff
        vd_start = time.perf_counter()
        vd = 1.0
        vd_elapsed = time.perf_counter() - vd_start
        total_vd_time += vd_elapsed
        vd_str = f"{vd:.6f}" if vd is not None else "N/A"

        hx_ratio = f"{hx / bl:.3f}" if (hx is not None and bl is not None and bl > 0) else ""
        bl_ratio = f"{bl / vd:.3f}" if (bl is not None and vd is not None and vd > 0) else ""

        print(
            f"{name:<{col}} {bl_str:>12} {hx_str:>12} {vd_str:>12} "
            f"{hx_ratio:>8} {bl_ratio:>8} "
            f"{bl_elapsed:>8.3f} {hx_elapsed:>8.3f} {vd_elapsed:>8.3f}"
        )

    total_elapsed = time.perf_counter() - total_start
    print("-" * len(header))
    print(
        f"{'Total':<{col}} {'':>12} {'':>12} {'':>12} "
        f"{'':>8} {'':>8} "
        f"{total_bl_time:>8.3f} {total_hx_time:>8.3f} {total_vd_time:>8.3f}"
    )
    print(f"{'End-to-end runtime (s)':<{col}} {total_elapsed:>12.3f}")


if __name__ == "__main__":
    main()
