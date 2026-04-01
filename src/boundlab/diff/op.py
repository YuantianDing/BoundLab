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

from boundlab.diff.expr import DiffExpr2


# =====================================================================
# Custom operator registration
# =====================================================================

_lib = torch.library.Library("boundlab", "DEF")
_lib.define("diff_pair(Tensor x, Tensor y) -> Tensor")

_lib.impl("diff_pair", lambda x, _: x, "CPU")
_lib.impl("diff_pair", lambda x, _: x, "CUDA")

# Shape/dtype inference for torch.export tracing (torch 2.x)
torch.library.impl_abstract("boundlab::diff_pair")(
    lambda x, _: torch.empty_like(x)
)


# =====================================================================
# Public API
# =====================================================================

def diff_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mark two tensors as a differentially-paired input.

    This is a registered :mod:`torch.library` custom operator, so it is
    captured verbatim when the containing model is exported with
    :func:`torch.export.export`.  At the concrete-tensor level it returns ``x``
    unchanged (a no-op).

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
    >>> any("diff_pair" in str(n.target) for n in gm.graph.nodes)
    True
    """
    return torch.ops.boundlab.diff_pair(x, y)


# =====================================================================
# Interpreter handler (used by boundlab.diff.zono3.interpret)
# =====================================================================

def diff_pair_handler(x, y) -> DiffExpr2:
    """Interpreter handler: convert a ``diff_pair`` node to a DiffExpr2.

    Registered in :data:`boundlab.diff.zono3.interpret` when this module is
    imported.
    """
    return DiffExpr2(x, y)
