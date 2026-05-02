"""Factory for unary differential linearizers backed by :mod:`boundlab.gradlin`.

Given a smooth 1D function ``f`` and a standard unary linearizer for the
single-variable bounds, build a differential linearizer that, for every
neuron, runs :func:`boundlab.gradlin.gradlin` on the batched trapezoid

    { (x, y) : x_lb <= x <= x_ub, y_lb <= y <= y_ub, d_lb <= x - y <= d_ub }

to tighten the linear bound on ``f(x) - f(y)``.
"""

from __future__ import annotations

from typing import Callable

import torch

from boundlab.expr._core import Expr
from boundlab.gradlin import gradlin
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds


def make_unary_diff_linearizer(
    f_1d: Callable[[torch.Tensor], torch.Tensor],
    std_linearizer: Callable[[torch.Tensor, torch.Tensor], ZonoBounds],
    *,
    iters: int = 30,
    lr: float = 0.1,
    name: str = "unary_diff_linearizer"
):
    """Build a ``(xs, ys, ds) -> DiffZonoBounds`` linearizer for ``f(x) − f(y)``.

    Parameters
    ----------
    f_1d
        Vectorised 1D function. Accepts a tensor of arbitrary shape and
        applies ``f`` element-wise.
    std_linearizer
        Standard (non-differential) zono linearizer with signature
        ``(ub, lb) -> ZonoBounds``; used for the per-branch ``x_bounds`` and
        ``y_bounds`` fields.
    iters, lr
        Forwarded to :func:`boundlab.gradlin.gradlin`.
    """

    def linearizer(xs: list[Expr], ys: list[Expr], ds: list[Expr]):
        from .. import DiffZonoBounds

        x, y, d = xs[0], ys[0], ds[0]
        x_ub, x_lb = ublb(x)
        y_ub, y_lb = ublb(y)
        d_ub, d_lb = ublb(d)
        shape = x_ub.shape
        ndim = len(shape)

        std_x = std_linearizer(x_ub, x_lb)
        std_y = std_linearizer(y_ub, y_lb)
        lam_x_std = std_x.input_weights[0].reshape(-1)
        lam_y_std = std_y.input_weights[0].reshape(-1)
        lam_init = torch.stack([lam_x_std, -lam_y_std], dim=-1)

        flat_lx = x_lb.reshape(-1)
        flat_ux = x_ub.reshape(-1)
        flat_ly = y_lb.reshape(-1)
        flat_uy = y_ub.reshape(-1)
        flat_ld = d_lb.reshape(-1)
        flat_ud = d_ub.reshape(-1)

        def F(x: torch.Tensor) -> torch.Tensor:
            return f_1d(x)

        lam_flat, L_flat, U_flat = gradlin(
            F,
            flat_lx,
            flat_ux,
            flat_ly,
            flat_uy,
            flat_ld,
            flat_ud,
            iters=iters,
            lr=lr,
            lam_init=lam_init,
        )

        lam_x = lam_flat[..., 0].reshape(shape)
        lam_y = lam_flat[..., 1].reshape(shape)
        L = L_flat.reshape(shape)
        U = U_flat.reshape(shape)

        bias = 0.5 * (L + U)
        err = 0.5 * (U - L)

        return DiffZonoBounds(
            x_bounds=std_x,
            y_bounds=std_y,
            diff_bounds=ZonoBounds(
                bias=bias,
                error_coeffs=EinsumOp.from_hardmard(err, ndim),
                input_weights=[0],
            ),
            diff_x_error=0,
            diff_x_weights=[lam_x],
            diff_y_error=0,
            diff_y_weights=[lam_y],
        )
    linearizer.__name__ = name
    return linearizer
