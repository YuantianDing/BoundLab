"""Heaviside handler registration for token pruning verification.

Registers ``HeavisidePruning`` in the standard zonotope interpreter so that
``heaviside_pruning(scores, data)`` ONNX nodes are handled during
verification.  The differential handler is already registered in the core
library (``boundlab.diff.zono3.default.heaviside``).

Import this module to activate the handler::

    import heaviside_handler  # registers at import time

No changes to the core BoundLab library are needed.
"""
from __future__ import annotations

import torch

import boundlab.zono as zono
from boundlab.expr._affine import ConstVal
from boundlab.expr._core import Expr
from boundlab.expr._var import LpEpsilon
from boundlab.diff.zono3.default.heaviside import _linearize_hsx


def heaviside_zono_handler(scores, data):
    """Standard zonotope handler for ``h(scores) * data``.

    With concrete scores (the common case during case-split verification),
    this collapses to an exact 0/1 mask with zero approximation error —
    mathematically identical to a concrete mask multiply.

    With symbolic scores (future work), falls back to the ``_linearize_hsx``
    affine relaxation from ``boundlab.diff.zono3.default.heaviside``.
    """
    # --- Fast path: concrete scores → exact mask, zero error ---
    if isinstance(scores, ConstVal):
        mask = (scores.value >= 0).float()
        return mask * data
    if isinstance(scores, torch.Tensor):
        mask = (scores >= 0).float()
        return mask * data

    # --- Symbolic scores → linearize h(s)*x ---
    if not isinstance(data, Expr):
        return NotImplemented

    s_ub, s_lb = scores.ublb()
    d_ub, d_lb = data.ublb()
    w_s, w_x, bias, err = _linearize_hsx(s_lb, s_ub, d_lb, d_ub)

    result = bias
    if (w_s.abs() > 0).any():
        result = result + w_s * scores
    if (w_x.abs() > 0).any():
        result = result + w_x * data
    if (err.abs() > 0).any():
        result = result + err * LpEpsilon(data.shape, reason="heaviside")

    return result


zono.interpret["HeavisidePruning"] = heaviside_zono_handler