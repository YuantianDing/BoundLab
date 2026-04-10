import onnx_ir
import torch
from torch import nn

import boundlab.expr as expr
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export


def _sample_inputs(center: torch.Tensor, scale: float, n: int = 2000) -> torch.Tensor:
    noise = torch.rand(n, *center.shape) * 2 - 1
    return center.unsqueeze(0) + scale * noise


def _check_bounds(outputs: torch.Tensor, ub: torch.Tensor, lb: torch.Tensor, tol: float = 1e-5):
    assert (outputs <= ub.unsqueeze(0) + tol).all(), (
        f"Upper bound violated: {(outputs - ub.unsqueeze(0)).max():.6f}"
    )
    assert (outputs >= lb.unsqueeze(0) - tol).all(), (
        f"Lower bound violated: {(lb.unsqueeze(0) - outputs).max():.6f}"
    )


class ReduceMeanModel(nn.Module):
    def forward(self, x):
        return x.mean(dim=-1, keepdim=True)


class ReduceSumModel(nn.Module):
    def forward(self, x):
        return x.sum(dim=(0, 2), keepdim=False)


def test_reduce_mean_sound():
    torch.manual_seed(0)
    model = ReduceMeanModel().eval()
    center = torch.randn(3, 4)
    scale = 0.2

    onnx_model = onnx_export(model, (center,))
    onnx_ir.save(onnx_model, "reduce_mean.onnx")
    assert any(node.op_type == "ReduceMean" for node in onnx_model.graph)

    op = zono.interpret(onnx_model)
    x = center + scale * expr.LpEpsilon(list(center.shape), p="inf")
    y = op(x)
    ub, lb = y.ublb()

    samples = _sample_inputs(center, scale, n=3000)
    outputs = samples.mean(dim=-1, keepdim=True)
    _check_bounds(outputs, ub, lb)


def test_reduce_sum_sound():
    torch.manual_seed(1)
    model = ReduceSumModel().eval()
    center = torch.randn(3, 4, 5)
    scale = 0.15

    onnx_model = onnx_export(model, (center,))
    assert any(node.op_type == "ReduceSum" for node in onnx_model.graph)

    op = zono.interpret(onnx_model)
    x = center + scale * expr.LpEpsilon(list(center.shape), p="inf")
    y = op(x)
    ub, lb = y.ublb()

    samples = _sample_inputs(center, scale, n=3000)
    outputs = samples.sum(dim=(1, 3))
    _check_bounds(outputs, ub, lb)
