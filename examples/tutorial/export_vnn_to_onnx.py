#!/usr/bin/env python3
"""Export a BoundLab VNN concretization pipeline to ONNX.

This example follows the same core flow used in ``zono_profile_heavy.py``:

1. Build a PyTorch model.
2. Wrap it with ``zono.interpret`` over symbolic input ``ConstVal(x) + eps``.
3. Concretize with ``ublb()``.
4. Export the resulting computation graph to ONNX.

The exported ONNX model takes a concrete input ``x`` and returns two outputs:
``ub`` and ``lb``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

import boundlab.expr as expr
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


class BoundPipeline(nn.Module):
    """Concrete-input wrapper that outputs symbolic bounds ``(ub, lb)``."""

    def __init__(self, model: nn.Module, input_dim: int, eps_scale: float) -> None:
        super().__init__()
        traced = torch.fx.symbolic_trace(model)
        self._op = zono.interpret(traced)
        self._input_dim = input_dim
        self._eps_scale = eps_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_expr = expr.ConstVal(x) + self._eps_scale * expr.LpEpsilon([self._input_dim])
        ub, lb = self._op(x_expr).ublb()
        return ub, lb


def export_bound_pipeline_to_onnx(
    *,
    output_path: Path,
    input_dim: int,
    width: int,
    depth: int,
    output_dim: int,
    eps_scale: float,
    seed: int,
) -> Path:
    """Build and export the bound pipeline to ONNX."""
    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    torch.manual_seed(seed)

    model = build_model(input_dim, width, depth, output_dim)
    pipeline = BoundPipeline(model, input_dim=input_dim, eps_scale=eps_scale)
    example_input = torch.randn(input_dim)

    # Export to a torch.export program first (helps symbolic shape support).
    shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
    with FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env):
        exported = torch.export.export(pipeline, args=(example_input,), strict=False)

    # Convert to ONNX.
    onnx_program = torch.onnx.export(exported, (example_input,), dynamo=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_program.save(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("./boundlab_vnn.onnx"), help="Output ONNX path.")
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--eps-scale", type=float, default=1.0, help="Scale for L-infinity perturbation symbol.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    onnx_path = export_bound_pipeline_to_onnx(
        output_path=args.output,
        input_dim=args.input_dim,
        width=args.width,
        depth=args.depth,
        output_dim=args.output_dim,
        eps_scale=args.eps_scale,
        seed=args.seed,
    )
    print(f"Exported ONNX model to: {onnx_path}")


if __name__ == "__main__":
    main()
