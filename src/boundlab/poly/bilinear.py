"""Bilinear operation handlers for polytope abstract interpretation."""

from __future__ import annotations

from typing import Union

import torch

from boundlab.expr._affine import ConstVal
from boundlab.expr._core import Expr
from boundlab.expr._var import LpEpsilon


def bilinear_elementwise(A: Expr, B: Expr) -> Expr:
    """Linearize element-wise product of two symbolic expressions."""
    assert A.shape == B.shape, \
        f"Shapes must match for element-wise product: {A.shape} vs {B.shape}"

    Ac, As = A.split_const()
    Bc, Bs = B.split_const()

    result = Ac * Bs + As * Bc + Ac * Bc

    if As.is_symmetric_to_0():
        Ahw = As.ub()
    else:
        A_ub, A_lb = As.ublb()
        Ahw = (A_ub - A_lb) / 2.0

    if Bs.is_symmetric_to_0():
        Bhw = Bs.ub()
    else:
        B_ub, B_lb = Bs.ublb()
        Bhw = (B_ub - B_lb) / 2.0

    error_bound = Ahw * Bhw
    new_eps = LpEpsilon(error_bound.shape)
    return result + error_bound * new_eps


def square_matmul(A: Expr, B: Expr) -> Expr:
    """Linearize ``A @ B`` using a square-split bilinear relaxation."""
    assert len(A.shape) >= 2 and len(B.shape) >= 2, \
        f"Need at least 2D for matmul, got {A.shape} @ {B.shape}"
    assert A.shape[:-2] == B.shape[:-2], \
        f"Batch dims must match: {A.shape} @ {B.shape}"
    assert A.shape[-1] == B.shape[-2], \
        f"Inner dims must match: {A.shape} @ {B.shape}"

    Ac, As = A.split_const()
    Bc, Bs = B.split_const()

    result = Ac @ Bs + As @ Bc + Ac @ Bc

    if As.is_symmetric_to_0() and Bs.is_symmetric_to_0():
        m, k, n = A.shape[-2], A.shape[-1], B.shape[-1]
        Au = As.ub()
        Bu = Bs.ub()
        U = Au @ Bu
        L = -U

        Au = Au.unsqueeze(-1).expand(*Au.shape, n)
        Bu = Bu.unsqueeze(-3).expand(*Bu.shape[:-2], m, k, n)
        As = As.unsqueeze(-1).expand(*As.shape, n)
        Bs = Bs.unsqueeze(-3).expand(*Bs.shape[:-2], m, k, n)

        a = torch.sqrt(Au)
        b = torch.sqrt(Bu)
        lama = a / b
        lamb = b / a
        Pos = torch.nan_to_num((lama * As + lamb * Bs).ub() ** 2 / 4, nan=1e10, posinf=1e10, neginf=1e10)
        Neg = torch.nan_to_num(-(lama * As - lamb * Bs).ub() ** 2 / 4, nan=-1e10, posinf=-1e10, neginf=-1e10)
        U = torch.minimum(Pos.sum(dim=-2), U)
        L = torch.maximum(Neg.sum(dim=-2), L)
    else:
        m, k, n = A.shape[-2], A.shape[-1], B.shape[-1]
        Au, Al = A.ublb()
        Bu, Bl = B.ublb()
        Ac = (Au + Al) / 2
        As = A - Ac
        Bc = (Bu + Bl) / 2
        Bs = B - Bc

        result = Ac @ Bc + As @ Bc + Ac @ Bs
        Asu = (Au - Al) / 2
        Bsu = (Bu - Bl) / 2
        U = Asu @ Bsu
        L = -U

        Asu = Asu.unsqueeze(-1).expand(*Asu.shape, n)
        Bsu = Bsu.unsqueeze(-3).expand(*Bsu.shape[:-2], m, k, n)
        As = As.unsqueeze(-1).expand(*As.shape, n)
        Bs = Bs.unsqueeze(-3).expand(*Bs.shape[:-2], m, k, n)

        a = torch.sqrt(Asu)
        b = torch.sqrt(Bsu)
        lama = a / b
        lamb = b / a
        PosU, PosL = (lama * As + lamb * Bs).ublb()
        NegU, NegL = (lama * As - lamb * Bs).ublb()
        Pos = torch.nan_to_num(torch.maximum(PosU ** 2, PosL ** 2) / 4, nan=1e10, posinf=1e10, neginf=1e10)
        Neg = torch.nan_to_num(-torch.maximum(NegU ** 2, NegL ** 2) / 4, nan=-1e10, posinf=-1e10, neginf=-1e10)
        U = torch.minimum(Pos.sum(dim=-2), U)
        L = torch.maximum(Neg.sum(dim=-2), L)

    result += (U + L) / 2 + (U - L) / 2 * LpEpsilon(result.shape)
    return result


def bilinear_matmul(A: Expr, B: Expr) -> Expr:
    """Linearize ``A @ B`` when both operands are symbolic expressions."""
    return square_matmul(A, B)


def matmul_handler(A, B):
    """Dispatcher implementation for ``torch.matmul``."""
    if isinstance(A, Expr) and _is_const(B):
        return A @ B
    if _is_const(A) and isinstance(B, Expr):
        return A @ B
    if _is_const(A) and _is_const(B):
        return torch.matmul(A, B)
    if isinstance(A, Expr) and isinstance(B, Expr):
        return bilinear_matmul(A, B)
    return NotImplemented


def mul_handler(A, B):
    """Dispatcher implementation for element-wise multiplication."""
    if isinstance(A, Expr) and _is_const(B):
        return _mul_expr_const(A, B)
    if _is_const(A) and isinstance(B, Expr):
        return _mul_expr_const(B, A)
    if _is_const(A) and _is_const(B):
        return torch.mul(A, B)
    if isinstance(A, Expr) and isinstance(B, Expr):
        return bilinear_elementwise(A, B)
    return NotImplemented


def _mul_expr_const(x: Expr, c):
    if isinstance(c, ConstVal):
        c = c.value

    if isinstance(c, (int, float)):
        return x * c
    if isinstance(c, torch.Tensor):
        if c.dim() == 0:
            return x * c
        out_shape = torch.broadcast_shapes(tuple(x.shape), tuple(c.shape))
        if tuple(x.shape) != out_shape:
            x = x.expand(*out_shape)
        if tuple(c.shape) != out_shape:
            c = c.expand(*out_shape)
        return x * c
    return x * c


def _is_const(tensor: Union[torch.Tensor, Expr, int]) -> bool:
    return isinstance(tensor, torch.Tensor) or isinstance(tensor, ConstVal) or isinstance(tensor, (int, float))
