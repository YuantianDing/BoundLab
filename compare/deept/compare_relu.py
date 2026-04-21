"""Compare BoundLab and DeepT zonotope bound propagation on simple MLPs.

Usage::

    python compare/deept/compare.py

Requires:
- opt_einsum  (pip install opt_einsum)
- termcolor   (pip install termcolor)
- DeepT submodule at compare/deept/DeepT/

For each test case the script propagates an L∞-norm-bounded input
perturbation through a randomly-initialised MLP and compares the
elementwise upper/lower bounds produced by each tool.

CPU feasibility
---------------
DeepT introduces one fresh error term per crossing neuron at every ReLU
layer, so its zonotope matrix can grow large quickly.  We skip DeepT when
the worst-case total error-term count (``input_dim + Σ hidden_sizes``)
multiplied by the final output width exceeds ``DEEPT_MAX_WORK``.
"""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

from boundlab.interp.onnx import onnx_export

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEEPT_DIR = HERE / "DeepT" / "Robustness-Verification-for-Transformers"

# ---------------------------------------------------------------------------
# BoundLab imports
# ---------------------------------------------------------------------------
import boundlab.expr as expr
import boundlab.zono as zono
from Zonotope import Zonotope as DeepTZonotope  # type: ignore
# ---------------------------a------------------------------------------------
# CPU feasibility threshold
# (max_error_terms * output_size elements in the final zonotope matrix)
# ---------------------------------------------------------------------------
DEEPT_MAX_WORK = 200_000


def _deept_feasible(input_dim: int, layer_sizes: list[int]) -> bool:
    """Return True when DeepT can plausibly finish on CPU in reasonable time.

    The worst-case total error terms after all ReLU layers equals
    ``input_dim + Σ hidden_sizes`` (every crossing neuron adds one term).
    Multiplied by the final output width this gives the dominant work unit.
    """
    hidden_sizes = layer_sizes[:-1]
    max_error_terms = input_dim + sum(hidden_sizes)
    return max_error_terms * layer_sizes[-1] <= DEEPT_MAX_WORK


# ---------------------------------------------------------------------------
# Minimal DeepT ``args`` Namespace
# ---------------------------------------------------------------------------
def _deept_args(input_dim: int, device: str = "cpu") -> Namespace:
    """Build the minimal Namespace expected by DeepT's Zonotope class."""
    return Namespace(
        perturbed_words=1,
        attack_type="l_inf",   # anything other than "synonym"
        all_words=False,
        device=device,
        cpu=(device == "cpu"),
        num_input_error_terms=input_dim,
        zonotope_slow=False,
        error_reduction_method="None",
        p=float("inf"),        # L∞ perturbation
    )


# ---------------------------------------------------------------------------
# BoundLab bound computation
# ---------------------------------------------------------------------------
def boundlab_bounds(
    model: nn.Sequential,
    center: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ub, lb) over the model output using BoundLab zonotope propagation.

    The input region is the L∞ ball ``{x : ‖x − center‖_∞ ≤ eps}``.
    """
    exported = onnx_export(model, (center,))
    op = zono.interpret(exported)
    noise = expr.LpEpsilon(list(center.shape), p="inf")
    x = center + (eps * noise)
    y = op(x)
    ub, lb = y.ublb()
    return ub, lb

# ---------------------------------------------------------------------------
# DeepT bound computation
# ---------------------------------------------------------------------------
def deept_bounds(
    model: nn.Sequential,
    center: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ub, lb) over the model output using DeepT's Zonotope class.

    The model must be a flat sequence of ``nn.Linear`` / ``nn.ReLU`` layers
    with a final ``nn.Linear`` (no trailing activation).
    """
    input_dim = center.shape[0]
    args = _deept_args(input_dim)

    # Build input zonotope: shape (1 + input_dim, 1, input_dim)
    # Each of the ``input_dim`` error terms has coefficient ``eps`` along its
    # own coordinate (standard L∞ box encoding).
    embeddings = center.reshape(1, -1)
    z = DeepTZonotope(args, p=float("inf"), eps=eps, perturbed_word_index=0, value=embeddings)

    # Forward pass through the network layers
    layers = list(model.children())
    for layer in layers:
        if isinstance(layer, nn.Linear):
            z = z.dense(layer)
        elif isinstance(layer, nn.ReLU):
            z = z.relu()
        else:
            raise ValueError(f"Unsupported layer type for DeepT comparison: {type(layer)}")

    lb, ub = z.concretize()
    # concretize returns tensors of shape (1, output_dim); squeeze to 1-D
    return ub.squeeze(0), lb.squeeze(0)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
# Each case: (name, input_dim, hidden_sizes, output_dim, eps)
CASES = [
    # ("Test   in=1  [2]→1",          4,  [8],       2, 0.1),
    ("Tiny   in=4  [8]→2",          4,  [8],       2, 0.1),
    ("Small  in=8  [16,16]→4",       8,  [16, 16],  4, 0.1),
    ("Medium in=16 [32,32]→4",      16,  [32, 32],  4, 0.05),
    ("Large  in=64 [128,128]→10",   64,  [128, 128], 10, 0.01),
    ("XLarge in=128 [256,256]→20", 128,  [256, 256], 20, 0.01),
]


def _build_model(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    """Build a ReLU MLP with the given architecture."""
    torch.manual_seed(42)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:



    col = 38
    w_bound = 10
    w_time = 9
    header = (
        f"{'Case':<{col}} {'BL ub[0]':>{w_bound}} {'BL lb[0]':>{w_bound}}"
        f" {'DT ub[0]':>{w_bound}} {'DT lb[0]':>{w_bound}}"
        f" {'BL(s)':>{w_time}} {'DT(s)':>{w_time}} {'DT/BL':>7} Note"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name, in_dim, hidden, out_dim, eps in CASES:
        layer_sizes = hidden + [out_dim]
        model = _build_model(in_dim, hidden, out_dim)
        center = torch.zeros(in_dim)

        # --- BoundLab ---
        t0 = time.perf_counter()
        bl_ub, bl_lb = boundlab_bounds(model, center, eps)
        bl_time = time.perf_counter() - t0

        # --- DeepT (only when CPU-feasible) ---
        feasible = _deept_feasible(in_dim, layer_sizes)
        if feasible:
            t0 = time.perf_counter()
            dt_ub, dt_lb = deept_bounds(model, center, eps)
            dt_time = time.perf_counter() - t0
            dt_ub0 = f"{dt_ub[0].item():.6f}"
            dt_lb0 = f"{dt_lb[0].item():.6f}"
            ratio = f"{dt_time / bl_time:.2f}x" if bl_time > 0 else "inf"
            note = ""
        else:
            dt_ub0 = dt_lb0 = "N/A"
            dt_time = float("nan")
            ratio = "N/A"
            note = "skipped (too large for CPU)"

        print(
            f"{name:<{col}}"
            f" {bl_ub[0].item():>{w_bound}.6f}"
            f" {bl_lb[0].item():>{w_bound}.6f}"
            f" {dt_ub0:>{w_bound}}"
            f" {dt_lb0:>{w_bound}}"
            f" {bl_time:>{w_time}.3f}"
            f" {(dt_time if feasible else float('nan')):>{w_time}.3f}"
            f" {ratio:>7}"
            f"  {note}"
        )

        if feasible:
            # Sanity check: BoundLab bounds should contain DeepT bounds
            # (both are over-approximations; neither is guaranteed tighter)
            bl_contains_dt = (bl_ub >= dt_ub - 1e-4).all() and (bl_lb <= dt_lb + 1e-4).all()
            dt_contains_bl = (dt_ub >= bl_ub - 1e-4).all() and (dt_lb <= bl_lb + 1e-4).all()
            if bl_contains_dt and not dt_contains_bl:
                print(f"  → DeepT is tighter on all outputs")
            elif dt_contains_bl and not bl_contains_dt:
                print(f"  → BoundLab is tighter on all outputs")
            elif bl_contains_dt and dt_contains_bl:
                print(f"  → Bounds are equal (within 1e-4)")
            else:
                print(f"  → Neither fully contains the other (mixed tightness)")

    print(sep)


if __name__ == "__main__":
    main()
