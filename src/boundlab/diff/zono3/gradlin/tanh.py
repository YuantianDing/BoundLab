"""Differential tanh linearizer tightened by :func:`boundlab.gradlin.gradlin`.

``sech²(x) = lam`` has two roots ``±atanh(√(1-lam))``; both are surfaced as
candidates so the optimiser considers positive- and negative-side critical
points on the same batch.
"""

from __future__ import annotations

import torch

from boundlab.zono.tanh import tanh_linearizer as _std_tanh_linearizer
from ._common import make_unary_diff_linearizer


def _tanh_grad_inv(lam: torch.Tensor) -> torch.Tensor:
    # tanh'(x) = 1 - tanh(x)² = lam ⇒ tanh(x) = ±√(1-lam).
    # Clamp: 1 − 1e-5 avoids fp32 rounding to 1.0 (which would blow up atanh).
    t = torch.sqrt((1.0 - lam).clamp(min=1e-5)).clamp(max=1 - 1e-5)
    pos = torch.atanh(t)
    return torch.stack([pos, -pos], dim=-1)  # (*batch, 2)


tanh_linearizer = make_unary_diff_linearizer(
    torch.tanh, _tanh_grad_inv, _std_tanh_linearizer
)
