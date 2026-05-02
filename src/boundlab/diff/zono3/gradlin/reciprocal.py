"""Differential reciprocal linearizer tightened by :func:`boundlab.gradlin.gradlin`.

Assumes both ``x`` and ``y`` are strictly positive. ``(1/x)' = -1/x²`` is
monotone on the positive axis, so ``grad_inv`` has a single branch.
"""

from __future__ import annotations

import torch

from boundlab.zono.reciprocal import reciprocal_linearizer as _std_reciprocal_linearizer
from ._common import make_unary_diff_linearizer


def _reciprocal_grad_inv(lam: torch.Tensor) -> torch.Tensor:
    # (1/x)' = -1/x² = lam  ⇒  x = 1 / √(-lam);  valid for lam < 0.
    safe = (-lam).clamp(min=1e-6)
    return 1.0 / torch.sqrt(safe)


reciprocal_linearizer = make_unary_diff_linearizer(
    torch.reciprocal, _std_reciprocal_linearizer, name="reciprocal_gradlin"
)
