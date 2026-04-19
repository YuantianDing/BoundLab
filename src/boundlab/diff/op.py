"""Operators for differential verification.

Registers ``boundlab::diff_pair`` as a :mod:`torch.library` custom operator so
that it can be captured by :func:`torch.export.export`.  The operator takes two
tensors of the same shape and returns a single *fake* tensor of the same shape
— a no-op at the concrete-tensor level whose sole purpose is to mark two
branches as a *paired* input for differential abstract interpretation.

When a :class:`~boundlab.interp.Interpreter` processes an exported graph, the
``diff_pair`` node is converted to a
:class:`~boundlab.diff.expr.DiffExpr2` by the registered handler.
"""

import torch
import torch.library
import onnxscript
from onnxscript import opset18 as _opset18

from boundlab.diff.zono3.expr import DiffExpr2
from boundlab.expr._affine import ConstVal

def diff_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mark two tensors as a differentially-paired input.

    This is a registered :mod:`torch.library` custom operator, so it is
    captured verbatim when the containing model is exported with
    :func:`torch.export.export`. During :func:`torch.onnx.export`, it is lowered
    to a custom-domain ONNX node ``boundlab::diff_pair``. At the
    concrete-tensor level it returns ``x`` unchanged (a no-op).

    When the exported graph is run through a differential interpreter
    (e.g. :data:`boundlab.diff.zono3.interpret`) the ``diff_pair`` node is
    replaced by a :class:`~boundlab.diff.expr.DiffExpr2` that tracks both
    branches simultaneously through all subsequent operations.

    Args:
        x: Tensor for the first network branch.
        y: Tensor for the second network branch; must have the same shape
           and dtype as ``x``.

    Returns:
        A fake tensor with the same shape and dtype as ``x``; carries no
        concrete information from ``y`` at runtime.

    Examples
    --------
    Exporting a model that uses ``diff_pair``:

    >>> import torch
    >>> from torch import nn
    >>> from boundlab.diff.op import diff_pair
    >>> class PairedModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(4, 3)
    ...     def forward(self, x, y):
    ...         p = diff_pair(x, y)
    ...         return self.fc(p)
    >>> model = PairedModel()
    >>> gm = torch.export.export(model, (torch.zeros(4), torch.zeros(4)))
    >>> any("DiffPair" in str(n.target) for n in gm.graph.nodes)
    True
    """
    return torch.onnx.ops.symbolic(
        "boundlab::DiffPair",
        (x, y),
        dtype=x.dtype,
        shape=x.shape,
        version=1,
    )


def heaviside_pruning(scores: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    A mock operator for differential verification that simulates the effect of pruning based on scores. This operator is not intended for actual use in production code, but serves as a placeholder for differential verification.

    For one network, this operator simply gives back the :code:`data` tensor.
    For the other network, it gives :code:`heaviside(scores) * data`, simulating the effect of pruning based on the scores.

    Args:
        scores: A tensor of scores used for pruning. Must have the same shape as :code
        data: A tensor of data to be pruned. Must have the same shape as :code:`scores`.

    Returns:
        A tensor of the same shape as :code:`scores` and :code:`data`.
    """
    assert scores.shape == data.shape[-scores.dim():], "scores and data must have the same shape"
    return torch.onnx.ops.symbolic(
        "boundlab::HeavisidePruning",
        (scores, data),
        dtype=data.dtype,
        shape=data.shape,
        version=1,
    )

def topk_pruning(scores: torch.Tensor, data: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    A mock operator for differential verification that simulates the effect of top-k pruning based on scores. This operator is not intended for actual use in production code, but serves as a placeholder for differential verification.

    For one network, this operator simply gives back the :code:`data` tensor.
    For the other network, it gives :code:`topk_mask(scores) * data`, simulating the effect of top-k pruning based on the scores.

    Args:
        scores: A tensor of scores used for pruning. Must have the same shape as :code:`data`.
        data: A tensor of data to be pruned. Must have the same shape as :code:`scores`.
        k: The number of top elements to keep based on the scores.
        dim: The dimension along which to compute the top-k elements.

    Returns:
        A tensor of the same shape as :code:`scores` and :code:`data`.
    """
    assert scores.shape == data.shape, "scores and data must have the same shape"
    dim = dim if dim >= 0 else scores.dim() + dim
    return torch.onnx.ops.symbolic(
        "boundlab::TopKPruning",
        (scores, data),
        attrs={"k": k, "dim": dim},
        dtype=data.dtype,
        shape=data.shape,
        version=1,
    )

import torch.nn as nn


class DiffLinear(nn.Module):
    """Two parallel linear layers whose outputs are paired via :func:`diff_pair`.

    At the concrete-tensor level this is equivalent to running ``fc1(x)``
    (``fc2``'s output is discarded at runtime via the ``diff_pair`` no-op).
    When the model is exported and interpreted by the differential interpreter
    (e.g. :data:`boundlab.diff.zono3.interpret`), the ``diff_pair`` node is
    lifted into a :class:`~boundlab.diff.expr.DiffExpr2` that tracks both
    branches simultaneously.

    Args:
        fc1: First linear layer.
        fc2: Second linear layer; must have the same ``in_features``,
             ``out_features``, and dtype as ``fc1``.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from boundlab.diff.op import DiffLinear
    >>> fc1 = nn.Linear(4, 3)
    >>> fc2 = nn.Linear(4, 3)
    >>> model = DiffLinear(fc1, fc2)
    >>> out = model(torch.zeros(4))
    >>> out.shape
    torch.Size([3])
    """

    def __init__(self, fc1: nn.Linear, fc2: nn.Linear):
        super().__init__()
        assert fc1.in_features == fc2.in_features, "fc1 and fc2 must share in_features"
        assert fc1.out_features == fc2.out_features, "fc1 and fc2 must share out_features"
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = diff_pair(self.fc1.weight, self.fc2.weight)
        assert (self.fc1.bias is not None) == (self.fc2.bias is not None), "fc1 and fc2 must both have bias or both have no bias"
        if self.fc1.bias is not None:
            return x @ weight.t() + diff_pair(self.fc1.bias, self.fc2.bias)
        else:
            return x @ weight.t()


# =====================================================================
# Interpreter handler (used by boundlab.diff.zono3.interpret)
# =====================================================================

def diff_pair_handler(x, y) -> DiffExpr2:
    """Interpreter handler: convert a ``diff_pair`` node to a DiffExpr2.

    Registered in :data:`boundlab.diff.zono3.interpret` when this module is
    imported.
    """
    if isinstance(x, torch.Tensor):
        x = ConstVal(x)
    if isinstance(y, torch.Tensor):
        y = ConstVal(y)
    return DiffExpr2(x, y)


__all__ = ["DiffPair", "DiffLinear", "HeavisidePruning", "TopKPruning"]
