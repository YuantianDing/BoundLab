"""Differential verification scaffolding for ``vit_threshold.py``.

Exposes :func:`diff_verify` — a helper that exports a ViT (or any module),
builds a :class:`DiffExpr3` around two input centers sharing an L-infinity
perturbation, and propagates it through
:data:`boundlab.diff.zono3.interpret` to bound the output difference
``net1(c1 ± eps) - net2(c2 ± eps)``.

``heaviside_pruning`` is *not yet* handled by the differential linearizer
registry (item 4 of ``vit_plan.md``).  We install an identity stub on import
so the rest of the model can be exercised end-to-end once the remaining
DiffExpr plumbing for ``Conv`` / ``Einsum`` lands.

A small unit-style model (:class:`TinyPrunedMLP`) is provided to exercise the
pipeline against today's ``diff.zono3`` capabilities (linear + relu + the
pruning stub).  Run ``python -m examples.vit.pruning`` to see it in action.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from boundlab import expr
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.diff.op import heaviside_pruning
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export


# ---------------------------------------------------------------------------
# Stub handler for ``heaviside_pruning`` (future work: item 4 in vit_plan.md)
# ---------------------------------------------------------------------------


def _heaviside_pruning_stub(scores, data):
    """Identity on ``data``; ``scores`` ignored.

    Placeholder until the real linearizer
    ``(x, y, d) -> (x, heaviside(scores_y) * y, ...)`` is implemented.
    """
    return data


diff_interpret["heaviside_pruning"] = _heaviside_pruning_stub


# ---------------------------------------------------------------------------
# Differential verification helper
# ---------------------------------------------------------------------------


def _zonotope(center: torch.Tensor, scale: float) -> expr.Expr:
    eps = expr.LpEpsilon(list(center.shape))
    return expr.ConstVal(center) + scale * eps


def diff_verify(
    model: nn.Module,
    c1: torch.Tensor,
    c2: torch.Tensor,
    scale: float,
) -> DiffExpr3 | DiffExpr2:
    """Bound ``model(c1 ± scale) - model(c2 ± scale)`` via zono3 interpretation."""
    assert c1.shape == c2.shape, "c1 and c2 must share shape"
    model.eval()
    onnx_model = onnx_export(model, (list(c1.shape),))
    op = diff_interpret(onnx_model)
    x = _zonotope(c1, scale)
    y = _zonotope(c2, scale)
    return op(DiffExpr3(x, y, x - y))


# ---------------------------------------------------------------------------
# Tiny model to exercise the pipeline end-to-end
# ---------------------------------------------------------------------------


class TinyPrunedMLP(nn.Module):
    """Minimal MLP using ``heaviside_pruning`` between hidden layers.

    Mirrors the shape of the ``vit_threshold`` pruning call so the stub
    handler is exercised on a real exported graph.
    """

    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.register_buffer("scores", torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = heaviside_pruning(self.scores, h)
        return self.fc2(h)


def _demo() -> None:
    torch.manual_seed(0)
    model = TinyPrunedMLP()
    c1 = torch.randn(8)
    c2 = c1 + 0.05 * torch.randn_like(c1)
    scale = 0.01

    out = diff_verify(model, c1, c2, scale)
    assert isinstance(out, DiffExpr3)
    d_ub, d_lb = out.diff.ublb()
    print("Diff bounds per class:")
    for i, (lb, ub) in enumerate(zip(d_lb.tolist(), d_ub.tolist())):
        print(f"  class {i}: [{lb:+.5f}, {ub:+.5f}]")
    print(f"Max diff width: {(d_ub - d_lb).max().item():.5f}")


if __name__ == "__main__":
    _demo()
