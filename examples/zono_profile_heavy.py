#!/usr/bin/env python3
"""Heavy zonotope workload for performance profiling.

Supports three propagation modes:
  interpret        – FX-traced abstract interpretation via the Expr system, exported to ONNX
  tensor           – pure tensor zonotope propagation (no Expr overhead)
  tensor_compiled  – same, but with torch.compile on the propagation function

Use ``--mode compare`` (default) to run interpret vs tensor,
check correctness, and print a performance summary.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import torch
from torch import nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono


def build_model(input_dim: int, width: int, depth: int, output_dim: int) -> nn.Module:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.ReLU())
        in_dim = width
    layers.append(nn.Linear(in_dim, output_dim))
    layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Mode 1: FX-traced abstract interpretation (original)
# ---------------------------------------------------------------------------

def run_workload(
    *,
    iters: int,
    input_dim: int,
    width: int,
    depth: int,
    output_dim: int,
    seed: int,
    clear_cache: bool,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor] | None]:
    import tempfile
    import onnxruntime as ort
    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    torch.manual_seed(seed)
    model = build_model(input_dim, width, depth, output_dim)
    traced = torch.fx.symbolic_trace(model)
    op = zono.interpret(traced)

    class Mod(torch.nn.Module):
        def forward(self, x):
            return op(expr.ConstVal(x) + expr.LpEpsilon([input_dim])).ublb()

    mod = Mod()
    example_input = torch.randn(input_dim)

    # Export to ONNX using dynamo export with ShapeEnv for SymInt support
    shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
    with FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env):
        exported = torch.export.export(mod, args=(example_input,), strict=False)

    # Convert exported program to ONNX and save to temp file
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    onnx_program = torch.onnx.export(exported, (example_input,), dynamo=True)
    onnx_program.save(onnx_path)
    print(f"Exported ONNX model to {onnx_path}")

    # Create ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    total_start = time.perf_counter()
    checksum = 0.0
    first_bounds = None

    # Reset seed before iteration loop for reproducible comparison
    torch.manual_seed(seed)

    for i in range(iters):
        x = torch.randn(input_dim)

        t0 = time.perf_counter()
        outputs = session.run(output_names, {input_name: x.numpy()})
        ub = torch.from_numpy(outputs[0])
        lb = torch.from_numpy(outputs[1])
        step_time = time.perf_counter() - t0

        if first_bounds is None:
            first_bounds = (ub.clone(), lb.clone())

        checksum += ub.abs().sum().item() + lb.abs().sum().item()
        print(f"iter {i + 1:02d}/{iters}: {step_time:.3f}s")

        if clear_cache:
            prop._UB_CACHE.clear()
            prop._LB_CACHE.clear()

    total_time = time.perf_counter() - total_start
    print(f"total: {total_time:.3f}s (checksum={checksum:.4f})")

    # Cleanup temp file
    import os
    os.unlink(onnx_path)

    return total_time, first_bounds


# ---------------------------------------------------------------------------
# Mode 2: Pure tensor zonotope propagation (no Expr system)
# ---------------------------------------------------------------------------

def run_tensor_workload(
    *,
    iters: int,
    input_dim: int,
    width: int,
    depth: int,
    output_dim: int,
    seed: int,
    clear_cache: bool,  # noqa: ARG001 – unused, kept for interface compat
) -> tuple[float, tuple[torch.Tensor, torch.Tensor] | None]:
    """Pure tensor zonotope propagation — no expression system.

    Represents the zonotope directly as (center, generators) where
    generators has shape (m, n) — m generators each of dimension n.
    The zonotope is: x = center + generators.T @ eps, eps ∈ [-1, 1]^m.

    Linear layers:  center' = W @ center + b,  generators' = generators @ W.T
    ReLU (triangle relaxation):
      - dead   (ub ≤ 0): zero out center and generators
      - active (lb ≥ 0): keep as-is
      - cross  (lb < 0 < ub): scale by slope = ub/(ub-lb), add bias and
        fresh error generators for the crossing neurons
    """
    torch.manual_seed(seed)
    model = build_model(input_dim, width, depth, output_dim)
    layers = list(model.children())

    total_start = time.perf_counter()
    checksum = 0.0
    first_bounds = None

    # Reset seed before iteration loop for reproducible comparison
    torch.manual_seed(seed)

    for i in range(iters):
        center = torch.randn(input_dim)
        generators = torch.eye(input_dim)  # (m, n): one generator per input dim

        t0 = time.perf_counter()
        with torch.no_grad():
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    center = center @ layer.weight.T + layer.bias
                    generators = generators @ layer.weight.T
                elif isinstance(layer, nn.ReLU):
                    gen_abs_sum = generators.abs().sum(dim=0)
                    lb_val = center - gen_abs_sum
                    ub_val = center + gen_abs_sum

                    dead = ub_val <= 0
                    active = lb_val >= 0
                    cross = ~dead & ~active

                    slope = torch.where(active, 1.0, 0.0)
                    slope = torch.where(cross, ub_val / (ub_val - lb_val), slope)

                    cross_val = torch.where(
                        cross,
                        -ub_val * lb_val / (2 * (ub_val - lb_val)),
                        0.0,
                    )

                    center = slope * center + cross_val
                    generators = generators * slope.unsqueeze(0)

                    # Add new error generators for crossing neurons
                    n_cross = int(cross.sum().item())
                    if n_cross > 0:
                        cross_idx = torch.nonzero(cross, as_tuple=False).squeeze(-1)
                        new_gens = torch.zeros(n_cross, center.shape[0])
                        new_gens[torch.arange(n_cross), cross_idx] = cross_val[cross_idx]
                        generators = torch.cat([generators, new_gens], dim=0)

            gen_abs_sum = generators.abs().sum(dim=0)
            ub = center + gen_abs_sum
            lb = center - gen_abs_sum

        step_time = time.perf_counter() - t0

        if first_bounds is None:
            first_bounds = (ub.clone(), lb.clone())

        checksum += ub.abs().sum().item() + lb.abs().sum().item()
        print(f"iter {i + 1:02d}/{iters}: {step_time:.3f}s")

    total_time = time.perf_counter() - total_start
    print(f"total: {total_time:.3f}s (checksum={checksum:.4f})")
    return total_time, first_bounds


# ---------------------------------------------------------------------------
# Mode 3: Compiled tensor zonotope propagation (torch.compile)
# ---------------------------------------------------------------------------

def _zonotope_forward(
    center: torch.Tensor,
    generators: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    biases: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.compile-friendly zonotope propagation.

    Each (weight, bias) pair represents a Linear layer followed by ReLU.
    To avoid data-dependent shapes (which prevent compilation), new error
    generators are always added as a full diagonal matrix (``torch.diag``).
    Zero entries for non-crossing neurons don't affect correctness.
    """
    for W, b in zip(weights, biases):
        # Linear
        center = center @ W.T + b
        generators = generators @ W.T
        # ReLU triangle relaxation
        gen_abs_sum = generators.abs().sum(dim=0)
        lb_val = center - gen_abs_sum
        ub_val = center + gen_abs_sum

        active = lb_val >= 0
        cross = (ub_val > 0) & ~active

        slope = torch.where(active, 1.0, 0.0)
        slope = torch.where(cross, ub_val / (ub_val - lb_val), slope)

        cross_val = torch.where(
            cross,
            -ub_val * lb_val / (2 * (ub_val - lb_val)),
            0.0,
        )

        center = slope * center + cross_val
        generators = generators * slope.unsqueeze(0)
        # Fixed-shape generator addition (avoids dynamic shapes)
        generators = torch.cat([generators, torch.diag(cross_val)], dim=0)

    gen_abs_sum = generators.abs().sum(dim=0)
    return center + gen_abs_sum, center - gen_abs_sum


def run_tensor_compiled_workload(
    *,
    iters: int,
    input_dim: int,
    width: int,
    depth: int,
    output_dim: int,
    seed: int,
    clear_cache: bool,  # noqa: ARG001
    compile_backend: str = "inductor",
) -> tuple[float, tuple[torch.Tensor, torch.Tensor] | None]:
    """Tensor zonotope propagation compiled with torch.compile."""
    torch.manual_seed(seed)
    model = build_model(input_dim, width, depth, output_dim)

    # Pre-extract weights and biases as tuples (paired Linear+ReLU)
    weights = tuple(
        m.weight.detach() for m in model.modules() if isinstance(m, nn.Linear)
    )
    biases = tuple(
        m.bias.detach() for m in model.modules() if isinstance(m, nn.Linear)
    )

    compiled_fn = torch.compile(_zonotope_forward, backend=compile_backend)

    total_start = time.perf_counter()
    checksum = 0.0
    first_bounds = None

    # Reset seed before iteration loop for reproducible comparison
    torch.manual_seed(seed)

    for i in range(iters):
        center = torch.randn(input_dim)
        generators = torch.eye(input_dim)

        t0 = time.perf_counter()
        with torch.no_grad():
            ub, lb = compiled_fn(center, generators, weights, biases)
        step_time = time.perf_counter() - t0

        if first_bounds is None:
            first_bounds = (ub.clone(), lb.clone())

        checksum += ub.abs().sum().item() + lb.abs().sum().item()
        print(f"iter {i + 1:02d}/{iters}: {step_time:.3f}s")

    total_time = time.perf_counter() - total_start
    print(f"total: {total_time:.3f}s (checksum={checksum:.4f})")
    return total_time, first_bounds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MODES = {
    "interpret": run_workload,
    "tensor": run_tensor_workload,
    "tensor_compiled": run_tensor_compiled_workload,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=5, help="Number of bound runs.")
    parser.add_argument("--input-dim", type=int, default=96 * 8, help="Input feature dimension.")
    parser.add_argument("--width", type=int, default=192 * 8, help="Hidden width.")
    parser.add_argument("--depth", type=int, default=5, help="Number of hidden ReLU layers.")
    parser.add_argument("--output-dim", type=int, default=64 * 8, help="Output dimension.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not clear global bound caches between iterations.",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=25,
        help="Print top functions by cumulative time. Set 0 to disable profiling.",
    )
    parser.add_argument(
        "--mode",
        choices=list(_MODES) + ["compare"],
        default="compare",
        help="Run a single mode or 'compare' all four (default: compare).",
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="torch.compile backend for compiled modes (default: inductor).",
    )
    args = parser.parse_args()

    clear_cache = not args.keep_cache
    common = dict(
        iters=args.iters,
        input_dim=args.input_dim,
        width=args.width,
        depth=args.depth,
        output_dim=args.output_dim,
        seed=args.seed,
        clear_cache=clear_cache,
    )
    compile_common = {**common, "compile_backend": args.compile_backend}

    if args.mode == "compare":
        mode_order = ("interpret", "tensor")
        results: dict[str, tuple[float, tuple[torch.Tensor, torch.Tensor]]] = {}
        for name in mode_order:
            print(f"\n{'=' * 20} {name} {'=' * 20}")
            t, bounds = _MODES[name](**common)
            results[name] = (t, bounds)

        # Correctness check against interpret baseline
        ref_ub, ref_lb = results["interpret"][1]
        print(f"\n{'=' * 20} correctness {'=' * 20}")
        for name in mode_order[1:]:
            ub, lb = results[name][1]
            ub_diff = (ub - ref_ub).abs().max().item()
            lb_diff = (lb - ref_lb).abs().max().item()
            print(f"{name:>20s} vs interpret:  ub_maxdiff={ub_diff:.2e}  lb_maxdiff={lb_diff:.2e}")

        # Performance summary
        ref_t = results["interpret"][0]
        print(f"\n{'=' * 20} performance {'=' * 20}")
        for name in mode_order:
            t = results[name][0]
            ratio = f"({t / ref_t:.2f}x)" if name != "interpret" else "(baseline)"
            print(f"{name:>20s}: {t:.3f}s  {ratio}")

    else:
        runner = _MODES[args.mode]
        kw = compile_common if "compiled" in args.mode else common
        if args.profile_top > 0:
            profiler = cProfile.Profile()
            profiler.enable()
            runner(**kw)
            profiler.disable()

            stats = pstats.Stats(profiler).sort_stats("cumtime")
            print(f"\nTop {args.profile_top} functions by cumulative time:")
            stats.print_stats(args.profile_top)
        else:
            runner(**kw)


if __name__ == "__main__":
    main()
