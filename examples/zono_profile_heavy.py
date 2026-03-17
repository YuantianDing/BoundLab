#!/usr/bin/env python3
"""Heavy zonotope workload for performance profiling."""

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


def run_workload(
    *,
    iters: int,
    input_dim: int,
    width: int,
    depth: int,
    output_dim: int,
    seed: int,
    clear_cache: bool,
) -> float:
    torch.manual_seed(seed)
    model = build_model(input_dim, width, depth, output_dim)
    traced = torch.fx.symbolic_trace(model)
    op = zono.interpret(traced)

    total_start = time.perf_counter()
    checksum = 0.0

    for i in range(iters):
        center = torch.randn(input_dim)
        x = expr.ConstVal(center) + expr.LpEpsilon([input_dim])

        t0 = time.perf_counter()
        y = op(x)
        ub, lb = y.ublb()
        step_time = time.perf_counter() - t0

        checksum += ub.abs().sum().item() + lb.abs().sum().item()
        print(f"iter {i + 1:02d}/{iters}: {step_time:.3f}s")

        if clear_cache:
            prop._UB_CACHE.clear()
            prop._LB_CACHE.clear()

    total_time = time.perf_counter() - total_start
    print(f"total: {total_time:.3f}s (checksum={checksum:.4f})")
    return total_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=12, help="Number of bound runs.")
    parser.add_argument("--input-dim", type=int, default=96, help="Input feature dimension.")
    parser.add_argument("--width", type=int, default=192, help="Hidden width.")
    parser.add_argument("--depth", type=int, default=5, help="Number of hidden ReLU layers.")
    parser.add_argument("--output-dim", type=int, default=64, help="Output dimension.")
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
    args = parser.parse_args()

    clear_cache = not args.keep_cache

    if args.profile_top > 0:
        profiler = cProfile.Profile()
        profiler.enable()
        run_workload(
            iters=args.iters,
            input_dim=args.input_dim,
            width=args.width,
            depth=args.depth,
            output_dim=args.output_dim,
            seed=args.seed,
            clear_cache=clear_cache,
        )
        profiler.disable()

        stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
        print(f"\nTop {args.profile_top} functions by cumulative time:")
        stats.print_stats(args.profile_top)
    else:
        run_workload(
            iters=args.iters,
            input_dim=args.input_dim,
            width=args.width,
            depth=args.depth,
            output_dim=args.output_dim,
            seed=args.seed,
            clear_cache=clear_cache,
        )


if __name__ == "__main__":
    main()
