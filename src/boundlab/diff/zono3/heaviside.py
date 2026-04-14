"""
Differential lineariser for ``heaviside_pruning``.

The ONNX op encodes a pruning mask applied only to the **y** network:

    out_x = x
    out_y = heaviside(scores_y) * y
    out_d = out_x - out_y

We rewrite the mask as ``y - h(-s_y) * y`` so the only non-linearity is the
product ``h(-s_y) * y``.  The helper below constructs an affine enclosure

    h(s) * x  ≈  w_s * s  + w_x * x  + bias  ± err

following the rules in ``examples/vit/vit_plan.md`` (item 4):

* If ``ls + us > 0`` linearise ``x - h(-s) * x`` instead (forces ``ls + us``
  non-positive).
* If ``lx + ux > 0`` linearise ``-h(s) * (-x)`` instead (forces ``lx + ux``
  non-positive).
* With ``ls + us <= 0`` and ``lx + ux <= 0``:
  - ``us < 0``: always zero.
  - ``ls < 0 < us`` and ``ux <= 0``: ``lam = max(lx / -ls, ux / us)`` with
    bounds ``lx <= h(s)*x - lam*s <= 0``.
  - ``ls < 0 < us`` and ``ux > 0``: ``lam = min(ux / (ux - lx), -lx / (ux - lx))``
    with bounds ``(1 - lam) * lx <= h(s)*x - lam*x <= (1 - lam) * ux``.

Both ``y`` and ``diff`` share the same approximation error epsilon so the mask
correlation is preserved between outputs.
"""

from __future__ import annotations

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._einsum import EinsumOp
from boundlab.prop import ublb
from boundlab.zono import ZonoBounds
from . import DiffZonoBounds, interpret


_EPS = 1e-30


def _linearize_base(ls, us, lx, ux):
    """Handles ls+us<=0 and lx+ux<=0; no flips."""
    zeros = torch.zeros_like(ls)
    w_s = zeros.clone()
    w_x = zeros.clone()
    bias = zeros.clone()
    err = zeros.clone()

    mask_zero = us < 0

    mask_case2 = (~mask_zero) & (ux <= 0)
    lam_num1 = torch.where(mask_case2, lx, zeros)
    lam_den1 = torch.where(mask_case2, -ls + _EPS, torch.ones_like(ls))
    lam_num2 = torch.where(mask_case2, ux, zeros)
    lam_den2 = torch.where(mask_case2, us + _EPS, torch.ones_like(us))
    lam_case2 = torch.maximum(lam_num1 / lam_den1, lam_num2 / lam_den2)
    w_s = torch.where(mask_case2, lam_case2, w_s)
    bias = torch.where(mask_case2, 0.5 * lx, bias)
    err = torch.where(mask_case2, -0.5 * lx, err)

    mask_case3 = (~mask_zero) & (ux > 0)
    denom3 = torch.where(mask_case3, ux - lx + _EPS, torch.ones_like(ux))
    lam3 = torch.minimum(ux / denom3, (-lx) / denom3)
    w_x = torch.where(mask_case3, lam3, w_x)
    bias = torch.where(mask_case3, 0.5 * (1 - lam3) * (lx + ux), bias)
    err = torch.where(mask_case3, 0.5 * (1 - lam3) * (ux - lx), err)

    return w_s, w_x, bias, err


def _linearize_no_s_flip(ls, us, lx, ux):
    """Assumes ls+us<=0; handles x-flip and base."""
    zeros = torch.zeros_like(ls)
    w_s = zeros.clone()
    w_x = zeros.clone()
    bias = zeros.clone()
    err = zeros.clone()

    mask_x_flip = lx + ux > 0
    ls2 = 
    ws_xflip, wx_xflip, b_xflip, e_xflip = _linearize_base(ls[mask_x_flip], us[mask_x_flip],
                                                           -ux[mask_x_flip], -lx[mask_x_flip])
    w_s[mask_x_flip] = -ws_xflip
    w_x[mask_x_flip] = wx_xflip
    bias[mask_x_flip] = -b_xflip
    err[mask_x_flip] = e_xflip

    mask_base = ~(mask_x_flip)
    ws_base, wx_base, b_base, e_base = _linearize_base(ls[mask_base], us[mask_base],
                                                       lx[mask_base], ux[mask_base])
    w_s[mask_base] = ws_base
    w_x[mask_base] = wx_base
    bias[mask_base] = b_base
    err[mask_base] = e_base

    return w_s, w_x, bias, err


def _linearize_hsx(ls: torch.Tensor, us: torch.Tensor,
                   lx: torch.Tensor, ux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (w_s, w_x, bias, err) such that

    ``h(s) * x ≈ w_s * s + w_x * x + bias ± err``.

    Element-wise implementation mirroring the case split in ``vit_plan.md``.
    """
    if ls.numel() == 0:
        z = torch.zeros_like(ls)
        return z, z, z, z

    zeros = torch.zeros_like(ls)
    w_s = zeros.clone()
    w_x = zeros.clone()
    bias = zeros.clone()
    err = zeros.clone()

    mask_s_flip = ls + us > 0
    ws_flip, wx_flip, b_flip, e_flip = _linearize_no_s_flip(
        -us[mask_s_flip], -ls[mask_s_flip], lx[mask_s_flip], ux[mask_s_flip]
    )
    w_s[mask_s_flip] = ws_flip
    w_x[mask_s_flip] = 1 - wx_flip
    bias[mask_s_flip] = -b_flip
    err[mask_s_flip] = e_flip

    mask_remaining = ~mask_s_flip
    ws_rest, wx_rest, b_rest, e_rest = _linearize_no_s_flip(
        ls[mask_remaining], us[mask_remaining], lx[mask_remaining], ux[mask_remaining]
    )
    w_s[mask_remaining] = ws_rest
    w_x[mask_remaining] = wx_rest
    bias[mask_remaining] = b_rest
    err[mask_remaining] = e_rest

    return w_s, w_x, bias, err


def _linearize_mask_term(s_y: Expr, y: Expr) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linearise ``h(-s_y) * y``.

    Returns weights for the original ``s_y`` (not ``-s_y``) and ``y``.
    """
    s_ub, s_lb = ublb(s_y)
    y_ub, y_lb = ublb(y)

    # We linearise h(s) * x with s = -s_y, x = y
    ls, us = -s_ub, -s_lb
    w_s, w_x, bias, err = _linearize_hsx(ls, us, y_lb, y_ub)

    # Convert back to sy (since s = -sy)
    w_sy = -w_s
    return w_sy, w_x, bias, err


def diff_heaviside_pruning_handler(scores, data):
    """Differential handler for ``boundlab::heaviside_pruning``.

    Args:
        scores: DiffExpr3 / DiffExpr2 providing pruning scores. Only the
                **y** component participates in masking.
        data:   DiffExpr3 / DiffExpr2 providing the tensor to prune.
    """
    from boundlab.diff.expr import DiffExpr2, DiffExpr3

    # Promote scores/data into DiffExpr3 when possible
    if isinstance(scores, DiffExpr2):
        scores = DiffExpr3(scores.x, scores.y, scores.x - scores.y)
    elif not isinstance(scores, DiffExpr3):
        # Treat constant scores as identical for both networks
        from boundlab import expr as _expr
        scores_expr = _expr.ConstVal(scores) if isinstance(scores, torch.Tensor) else scores
        scores = DiffExpr3(scores_expr, scores_expr, scores_expr * 0)

    if isinstance(data, DiffExpr2):
        data = DiffExpr3(data.x, data.y, data.x - data.y)
    elif not isinstance(data, DiffExpr3):
        from boundlab import expr as _expr
        data_expr = _expr.ConstVal(data) if isinstance(data, torch.Tensor) else data
        data = DiffExpr3(data_expr, data_expr, data_expr * 0)

    assert isinstance(scores, DiffExpr3) and isinstance(data, DiffExpr3), (
        "heaviside_pruning requires expressions convertible to DiffExpr3")

    sy = scores.y
    y = data.y

    # Linearise t = h(-sy) * y  →  weights on sy and y
    w_sy, w_y, bias_t, err_t = _linearize_mask_term(sy, y)

    zeros = torch.zeros_like(bias_t)

    # x component: passthrough of data.x
    x_bounds = ZonoBounds(
        bias=zeros,
        error_coeffs=zeros,
        input_weights=[0, torch.ones_like(w_y)],
    )

    # y component: y - t
    y_bounds = ZonoBounds(
        bias=-bias_t,
        error_coeffs=err_t,
        input_weights=[w_sy, torch.ones_like(w_y) - w_y],
    )

    # diff component: d + t
    diff_bounds = ZonoBounds(
        bias=bias_t,
        error_coeffs=zeros,
        input_weights=[0, torch.ones_like(w_y)],  # weights on ds: (scores_diff, data_diff)
    )

    err_op = EinsumOp.from_hardmard(err_t, len(err_t.shape))

    dzb = DiffZonoBounds(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        diff_bounds=diff_bounds,
        diff_x_error=EinsumOp.from_hardmard(zeros, len(err_t.shape)),
        diff_x_weights=[0, 0],
        diff_y_error=err_op,
        diff_y_weights=[-w_sy, w_y],
    )

    from . import _build_triple_from_dzb
    xs = [scores.x, data.x]
    ys = [scores.y, data.y]
    ds = [scores.diff, data.diff]
    return _build_triple_from_dzb(dzb, xs, ys, ds)


# Register handler
interpret["heaviside_pruning"] = diff_heaviside_pruning_handler


__all__ = ["diff_heaviside_pruning_handler"]
