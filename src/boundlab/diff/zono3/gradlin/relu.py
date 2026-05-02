"""Differential ReLU linearizer tightened by :func:`boundlab.gradlin.gradlin`.

This uses the same trapezoid search pipeline as the smooth activations, but
with the unary function ``relu`` instead of a smooth derivative inverse.
"""

from __future__ import annotations

import torch

from boundlab.expr._core import Expr
from boundlab.gradlin import gradlin
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from boundlab.zono.relu import relu_linearizer as _std_relu_linearizer


_ITERS = 100
_LR = 0.1


def relu_linearizer(xs: list[Expr], ys: list[Expr], ds: list[Expr]):
    from .. import DiffZonoBounds

    x, y, d = xs[0], ys[0], ds[0]
    x_ub, x_lb = ublb(x)
    y_ub, y_lb = ublb(y)
    d_ub, d_lb = ublb(d)
    shape = x_ub.shape
    ndim = len(shape)

    std_x = _std_relu_linearizer(x_ub, x_lb)
    std_y = _std_relu_linearizer(y_ub, y_lb)
    lam_init = torch.stack(
        [std_x.input_weights[0].reshape(-1), -std_y.input_weights[0].reshape(-1)],
        dim=-1,
    )

    lam_flat, L_flat, U_flat = gradlin(
        torch.relu,
        x_lb.reshape(-1),
        x_ub.reshape(-1),
        y_lb.reshape(-1),
        y_ub.reshape(-1),
        d_lb.reshape(-1),
        d_ub.reshape(-1),
        iters=_ITERS,
        lr=_LR,
        lam_init=lam_init,
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
