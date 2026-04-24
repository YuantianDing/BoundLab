"""Differential ReLU linearizer tightened by :func:`boundlab.gradlin.gradlin`.

ReLU has no smooth inverse gradient — instead it has piecewise-linear kinks
along ``x = 0`` and ``y = 0``. Extrema of ``relu(x) − relu(y) − lam·(x, y)``
on the trapezoidal region therefore include, in addition to the polytope
vertices, the endpoints of the line segments ``R ∩ {x = 0}`` and
``R ∩ {y = 0}``. We surface those endpoints as fixed candidates to
``gradlin`` (via its ``grad_inv`` hook — ``gradlin`` checks each candidate
for feasibility inside ``R``) so the optimiser sees every place where the
residual could peak.
"""

from __future__ import annotations

import torch

from boundlab.expr._core import Expr
from boundlab.gradlin import gradlin, trapezoid_region
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.relu import relu_linearizer as _std_relu_linearizer


_ITERS = 100
_LR = 0.1


def _F(xy: torch.Tensor) -> torch.Tensor:
    return torch.relu(xy[..., 0]) - torch.relu(xy[..., 1])


def _make_grad_inv(d_lb: torch.Tensor, d_ub: torch.Tensor):
    """Return the relu-kink candidate set ignoring ``lam``.

    Candidates are ``(0,0)`` plus the four points where the kink lines
    ``x = 0`` and ``y = 0`` meet the diff-constraint edges ``x − y = ld``
    and ``x − y = ud``. Combined with gradlin's built-in axis-face
    coordinate-replacement this covers all the kink-line endpoints on the
    trapezoid's boundary.
    """

    def grad_inv(lam: torch.Tensor) -> torch.Tensor:
        batch = lam.shape[:-1]
        dl = d_lb.expand(batch)
        du = d_ub.expand(batch)
        zeros = torch.zeros_like(dl)
        return torch.stack(
            [
                torch.stack([zeros, zeros], dim=-1),
                torch.stack([zeros, -du], dim=-1),
                torch.stack([zeros, -dl], dim=-1),
                torch.stack([du, zeros], dim=-1),
                torch.stack([dl, zeros], dim=-1),
            ],
            dim=-2,
        )  # (*batch, 5, 2)

    return grad_inv


def relu_linearizer(xs: list[Expr], ys: list[Expr], ds: list[Expr]):
    from .. import DiffZonoBounds

    x, y, d = xs[0], ys[0], ds[0]
    x_ub, x_lb = ublb(x)
    y_ub, y_lb = ublb(y)
    d_ub, d_lb = ublb(d)
    shape = x_ub.shape
    ndim = len(shape)

    lb, ub, A_extra, b_extra = trapezoid_region(
        x_lb.reshape(-1), x_ub.reshape(-1),
        y_lb.reshape(-1), y_ub.reshape(-1),
        d_lb.reshape(-1), d_ub.reshape(-1),
    )

    grad_inv = _make_grad_inv(d_lb.reshape(-1), d_ub.reshape(-1))

    std_x = _std_relu_linearizer(x_ub, x_lb)
    std_y = _std_relu_linearizer(y_ub, y_lb)
    lam_init = torch.stack(
        [std_x.input_weights[0].reshape(-1), -std_y.input_weights[0].reshape(-1)],
        dim=-1,
    )

    lam_flat, L_flat, U_flat = gradlin(
        _F, grad_inv, lb, ub, A_extra, b_extra,
        iters=_ITERS, lr=_LR, lam_init=lam_init,
    )

    lam_x = lam_flat[..., 0].reshape(shape)
    lam_y = lam_flat[..., 1].reshape(shape)
    L = L_flat.reshape(shape)
    U = U_flat.reshape(shape)

    lam_avg = 0.5 * (lam_x + lam_y)
    lam_x = lam_x - lam_avg
    lam_y = lam_avg - lam_y

    bias = 0.5 * (L + U)
    err = 0.5 * (U - L)


    return DiffZonoBounds(
        x_bounds=std_x,
        y_bounds=std_y,
        diff_bounds=ZonoBounds(
            bias=bias,
            error_coeffs=EinsumOp.from_hardmard(err, ndim),
            input_weights=[lam_avg],
        ),
        diff_x_error=0,
        diff_x_weights=[lam_x],
        diff_y_error=0,
        diff_y_weights=[lam_y],
    )
