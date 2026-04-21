#!/usr/bin/env python3
"""Heavy transformer workload for performance profiling.

Profiles zonotope abstract interpretation on transformer-like architectures:
  linear    – single-head linear attention (bilinear matmul, no softmax) + FFN
  softmax   – single-head softmax attention (exp + reciprocal + bilinear) + FFN

Run with pyinstrument (default) or cProfile:

    python examples/transformer_profile_heavy.py --mode linear
    python examples/transformer_profile_heavy.py --mode softmax --profiler cprofile
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class LinearAttentionBlock(nn.Module):
    """Single-head linear attention (no softmax) + ReLU FFN."""

    def __init__(self, d_model: int, d_k: int, d_ff: int):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        self.proj = nn.Linear(d_k, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        attn = torch.matmul(Q, K.transpose(0, 1))
        context = torch.matmul(attn, V)
        out = self.proj(context)
        x = x + out
        h = self.relu(self.ff1(x))
        return x + self.ff2(h)


class SoftmaxAttentionBlock(nn.Module):
    """Single-head softmax attention + ReLU FFN."""

    def __init__(self, d_model: int, d_k: int, d_ff: int):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        self.proj = nn.Linear(d_k, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_ff, d_model)
        self.scale = math.sqrt(d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        out = self.proj(context)
        x = x + out
        h = self.relu(self.ff1(x))
        return x + self.ff2(h)


_MODELS = {
    "linear": LinearAttentionBlock,
    "softmax": SoftmaxAttentionBlock,
}


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

def run_workload(
    *,
    mode: str,
    iters: int,
    seq_len: int,
    d_model: int,
    d_k: int,
    d_ff: int,
    scale: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    model_cls = _MODELS[mode]
    model = model_cls(d_model, d_k, d_ff)
    model.eval()

    onnx_model = onnx_export(model, ([seq_len, d_model],))
    op = zono.interpret(onnx_model)

    total_start = time.perf_counter()
    checksum = 0.0

    torch.manual_seed(seed)
    for i in range(iters):
        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()

        center_val = torch.randn(seq_len, d_model) * 0.5
        center = expr.ConstVal(center_val)
        eps = expr.LpEpsilon([seq_len, d_model])
        x_expr = center + eps * scale

        t0 = time.perf_counter()
        y_expr = op(x_expr)
        ub, lb = y_expr.ublb()
        step_time = time.perf_counter() - t0

        checksum += ub.abs().sum().item() + lb.abs().sum().item()
        print(f"iter {i + 1:02d}/{iters}: {step_time:.3f}s")

    total_time = time.perf_counter() - total_start
    print(f"total: {total_time:.3f}s (checksum={checksum:.4f})")
    return total_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=list(_MODELS),
        default="linear",
        help="Transformer variant to profile (default: linear).",
    )
    parser.add_argument("--iters", type=int, default=3, help="Number of bound runs.")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length.")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension.")
    parser.add_argument("--d-k", type=int, default=64, help="Key/query dimension.")
    parser.add_argument("--d-ff", type=int, default=128, help="FFN hidden dimension.")
    parser.add_argument("--scale", type=float, default=0.01, help="Perturbation scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--profiler",
        choices=["pyinstrument", "cprofile", "none"],
        default="pyinstrument",
        help="Profiler to use (default: pyinstrument).",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=25,
        help="Top functions to show for cProfile (default: 25).",
    )
    args = parser.parse_args()

    common = dict(
        mode=args.mode,
        iters=args.iters,
        seq_len=args.seq_len,
        d_model=args.d_model,
        d_k=args.d_k,
        d_ff=args.d_ff,
        scale=args.scale,
        seed=args.seed,
    )

    if args.profiler == "pyinstrument":
        from pyinstrument import Profiler

        profiler = Profiler()
        with profiler:
            run_workload(**common)
        profiler.open_in_browser()

    elif args.profiler == "cprofile":
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        run_workload(**common)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        print(f"\nTop {args.profile_top} functions by cumulative time:")
        stats.print_stats(args.profile_top)

    else:
        run_workload(**common)


if __name__ == "__main__":
    main()
