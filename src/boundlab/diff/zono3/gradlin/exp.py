"""Differential exp linearizer tightened by :func:`boundlab.gradlin.gradlin`.

On the 2D trapezoid ``{(x,y) : x_lb≤x≤x_ub, y_lb≤y≤y_ub, d_lb≤x-y≤d_ub}``
we linearly bound ``exp(x) − exp(y)``. Unlike the closed-form default, the
slopes on ``x`` and ``y`` are *decoupled* — this helps when the ``x``- and
``y``-ranges are asymmetric.
"""

from __future__ import annotations

import torch

from boundlab.zono.exp import exp_linearizer as _std_exp_linearizer
from ._common import make_unary_diff_linearizer


def _exp_grad_inv(lam: torch.Tensor) -> torch.Tensor:
    # exp'(x) = exp(x) = lam ⇒ x = log(lam); valid for lam > 0.
    safe = lam.clamp(min=1e-6)
    return torch.log(safe)


exp_linearizer = make_unary_diff_linearizer(
    torch.exp, _exp_grad_inv, _std_exp_linearizer
)
