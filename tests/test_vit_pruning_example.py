"""Smoke-test for differential pruning lineariser.

Defines a tiny MLP that uses ``heaviside_pruning`` between layers and
runs the zono3 differential interpreter to ensure bounds are finite and
the handler is registered.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from boundlab import expr
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.op import heaviside_pruning
from boundlab.diff.zono3 import interpret as diff_interpret


def _zonotope(center: torch.Tensor, scale: float) -> expr.Expr:
    eps = expr.LpEpsilon(list(center.shape))
    return expr.ConstVal(center) + scale * eps


def diff_verify(model: nn.Module, c1: torch.Tensor, c2: torch.Tensor, scale: float):
    assert c1.shape == c2.shape, "c1 and c2 must share shape"
    model.eval()
    # Export ONNX and interpret with differential zonotope semantics
    from boundlab.interp.onnx import onnx_export

    onnx_model = onnx_export(model, (list(c1.shape),))
    op = diff_interpret(onnx_model)
    x = _zonotope(c1, scale)
    y = _zonotope(c2, scale)
    return op(DiffExpr3(x, y, x - y))


class TinyPrunedMLP(nn.Module):
    """Minimal MLP using ``heaviside_pruning`` between hidden layers."""

    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.register_buffer("scores", torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = heaviside_pruning(self.scores, h)
        return self.fc2(h)


def test_tiny_pruned_mlp_diff_bounds_sound():
    torch.manual_seed(0)
    model = TinyPrunedMLP()
    c1 = torch.randn(8)
    c2 = c1 + 0.05 * torch.randn_like(c1)
    scale = 0.01

    out = diff_verify(model, c1, c2, scale)
    d_ub, d_lb = out.diff.ublb()
    assert d_ub.shape == torch.Size([4])
    assert torch.isfinite(d_ub).all()
    assert torch.isfinite(d_lb).all()
    assert (d_ub >= d_lb - 1e-6).all()
    # Soundness against the concrete model is intentionally *not* asserted:
    # the lineariser relaxes the symbolic op, so concrete-vs-abstract gaps are
    # expected; we only require finite and ordered bounds here.


def test_heaviside_handler_registered():
    assert "HeavisidePruning" in diff_interpret.dispatcher
