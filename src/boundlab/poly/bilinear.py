"""Bilinear matmul handler for poly abstract interpretation."""

from __future__ import annotations

import torch

from boundlab.expr._affine import ConstVal
from boundlab.expr._core import Expr
from .square import square_linearizer


def _square_expr(x: Expr) -> Expr:
    from . import _bounds_to_expr

    ub, lb = x.ublb()
    return _bounds_to_expr(x, square_linearizer(ub, lb))


def bilinear_elementwise(A: Expr, B: Expr) -> Expr:
    """Linearize elementwise product using (x+y)^2/4 - (x-y)^2/4."""
    assert A.shape == B.shape, \
        f"Shapes must match for element-wise product: {A.shape} vs {B.shape}"
    return 0.25 * (_square_expr(A + B) - _square_expr(A - B))


def _symbolic_matmul(A: Expr, B: Expr) -> Expr:
    """Linearize symbolic ``A @ B`` by summing square-split products."""
    assert len(A.shape) >= 2 and len(B.shape) >= 2, \
        f"Need at least 2D for matmul, got {A.shape} @ {B.shape}"
    assert A.shape[:-2] == B.shape[:-2], \
        f"Batch dims must match: {A.shape} @ {B.shape}"
    assert A.shape[-1] == B.shape[-2], \
        f"Inner dims must match: {A.shape} @ {B.shape}"

    *batch, m, k = A.shape
    n = B.shape[-1]
    out = None
    a_dim_k = len(A.shape) - 1
    b_dim_k = len(B.shape) - 2
    for i in range(k):
        a_i = A.narrow(a_dim_k, i, 1).expand(*batch, m, n)
        b_i = B.narrow(b_dim_k, i, 1).expand(*batch, m, n)
        term = bilinear_elementwise(a_i, b_i)
        out = term if out is None else out + term
    return out


def bilinear_matmul(A: Expr, B: Expr) -> Expr:
    """Linearize ``A @ B`` when both operands are symbolic expressions."""
    Ac, As = A.split_const()
    Bc, Bs = B.split_const()
    return Ac @ Bs + As @ Bc + Ac @ Bc + _symbolic_matmul(As, Bs)


def matmul_handler(A, B):
    if isinstance(A, Expr) and _is_const(B):
        return A @ B
    if _is_const(A) and isinstance(B, Expr):
        return A @ B
    if _is_const(A) and _is_const(B):
        return torch.matmul(A, B)
    if isinstance(A, Expr) and isinstance(B, Expr):
        return bilinear_matmul(A, B)
    return A @ B


def mul_handler(A, B):
    if isinstance(A, Expr) and _is_const(B):
        return _mul_expr_const(A, B)
    if _is_const(A) and isinstance(B, Expr):
        return _mul_expr_const(B, A)
    if _is_const(A) and _is_const(B):
        return torch.mul(A, B)
    if isinstance(A, Expr) and isinstance(B, Expr):
        return bilinear_elementwise(A, B)
    return A * B


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


def _is_const(tensor) -> bool:
    return isinstance(tensor, torch.Tensor) or isinstance(tensor, ConstVal) or isinstance(tensor, (int, float))
