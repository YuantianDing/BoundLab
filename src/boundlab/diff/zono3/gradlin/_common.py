"""Factory for unary differential linearizers backed by :mod:`boundlab.gradlin`.

Given a smooth 1D function ``f`` and a multi-branch inverse ``grad_inv_1d``
(roots of ``f'(x) = c``), build a differential linearizer that, for every
neuron, runs :func:`boundlab.gradlin.gradlin` on the 2D trapezoid
``{(x, y) : x_lb ≤ x ≤ x_ub, y_lb ≤ y ≤ y_ub, d_lb ≤ x − y ≤ d_ub}`` to
tighten the linear bound on ``f(x) − f(y)``.

The optimiser runs once per layer (batched across all neurons) and costs
``iters`` Adam steps; use this module when the closed-form linearizers in
:mod:`boundlab.diff.zono3.default` leave too much slack.
"""

from __future__ import annotations

from typing import Callable

import torch

from boundlab.expr._core import Expr
from boundlab.gradlin import gradlin, trapezoid_region
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds


def make_unary_diff_linearizer(
    f_1d: Callable[[torch.Tensor], torch.Tensor],
    grad_inv_1d: Callable[[torch.Tensor], torch.Tensor],
    std_linearizer: Callable[[torch.Tensor, torch.Tensor], ZonoBounds],
    *,
    iters: int = 100,
    lr: float = 0.1,
):
    """Build a ``(xs, ys, ds) -> DiffZonoBounds`` linearizer for ``f(x) − f(y)``.

    Parameters
    ----------
    f_1d
        Vectorised 1D function. Accepts a tensor of arbitrary shape and
        applies ``f`` element-wise.
    grad_inv_1d
        Inverse derivative. Accepts ``lam`` of shape ``(*batch,)`` and
        returns candidate roots of ``f'(x) = lam`` with shape ``(*batch, K)``
        — multiple branches are allowed (clamp out-of-domain values; the
        feasibility mask inside :mod:`boundlab.gradlin` filters them out).
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

        lb, ub, A_extra, b_extra = trapezoid_region(
            flat_lx, flat_ux, flat_ly, flat_uy, flat_ld, flat_ud
        )

        def F(xy: torch.Tensor) -> torch.Tensor:
            return f_1d(xy[..., 0]) - f_1d(xy[..., 1])

        def grad_inv_2d(lam: torch.Tensor) -> torch.Tensor:
            cand_x = grad_inv_1d(lam[..., 0])        # (*batch, Kx)
            cand_y = grad_inv_1d(-lam[..., 1])       # (*batch, Ky)
            if cand_x.dim() == lam.dim() - 1:
                cand_x = cand_x.unsqueeze(-1)
            if cand_y.dim() == lam.dim() - 1:
                cand_y = cand_y.unsqueeze(-1)
            Kx = cand_x.shape[-1]
            Ky = cand_y.shape[-1]
            cx = cand_x.unsqueeze(-1).expand(*cand_x.shape, Ky)      # (*batch, Kx, Ky)
            cy = cand_y.unsqueeze(-2).expand(*cand_y.shape[:-1], Kx, Ky)
            stacked = torch.stack([cx, cy], dim=-1)                  # (*batch, Kx, Ky, 2)
            return stacked.flatten(-3, -2)                           # (*batch, Kx*Ky, 2)

        lam_flat, L_flat, U_flat = gradlin(
            F, grad_inv_2d, lb, ub, A_extra, b_extra,
            iters=iters, lr=lr, lam_init=lam_init,
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

    return linearizer
