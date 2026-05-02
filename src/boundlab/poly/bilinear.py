"""Bilinear matmul handler for poly abstract interpretation."""

from __future__ import annotations

import torch

from boundlab.expr._affine import ConstVal
from boundlab.expr._core import Expr
from boundlab.expr._var import LpEpsilon
from .square import square_linearizer


def _square_expr(x: Expr) -> Expr:
    from . import _bounds_to_expr

    ub, lb = x.ublb()
    return _bounds_to_expr(x, square_linearizer(ub, lb), reason=square_linearizer.__name__)


def bilinear_elementwise(A: Expr, B: Expr) -> Expr:
    """Linearize elementwise product using (x+y)^2/4 - (x-y)^2/4."""
    assert A.shape == B.shape, \
        f"Shapes must match for element-wise product: {A.shape} vs {B.shape}"
    return 0.25 * (_square_expr(A + B) - _square_expr(A - B))


def square_matmul(A: Expr, B: Expr) -> Expr:
    """Linearize ``A @ B`` using the optimal-λ square-split bound.

    For each (m, k, n) position, uses the identity
    ``As[m,k] * Bs[k,n] = ((λAs + λ⁻¹Bs)² - (λAs - λ⁻¹Bs)²) / 4``
    with ``λ = sqrt(|Au| / |Bu|)`` to produce a tighter bound than the
    naive McCormick ``|Au| @ |Bu|``. Unlike the zono version, symmetry
    around zero is not assumed; ``max(|ub|, |lb|)² / 4`` is used so the
    formula remains valid for asymmetric expressions.
    """
    assert len(A.shape) >= 2 and len(B.shape) >= 2, \
        f"Need at least 2D for matmul, got {A.shape} @ {B.shape}"
    assert A.shape[:-2] == B.shape[:-2], \
        f"Batch dims must match: {A.shape} @ {B.shape}"
    assert A.shape[-1] == B.shape[-2], \
        f"Inner dims must match: {A.shape} @ {B.shape}"

    Au, Al = A.ublb()
    Bu, Bl = B.ublb()
    Ac = (Au + Al) / 2
    As = (Au - Al) / 2
    
    result = Ac @ B + As @ Bc + Ac @ Bc

    m, k, n = A.shape[-2], A.shape[-1], B.shape[-1]
    Au_abs = torch.maximum(As.ub().abs(), As.lb().abs())  # (..., m, k)
    Bu_abs = torch.maximum(Bs.ub().abs(), Bs.lb().abs())  # (..., k, n)

    # Naive element-wise absolute bound, then sum over k.
    Au_exp = Au_abs.unsqueeze(-1).expand(*Au_abs.shape, n)       # (..., m, k, n)
    Bu_exp = Bu_abs.unsqueeze(-3).expand(*Bu_abs.shape[:-2], m, k, n)  # (..., m, k, n)
    U = (Au_exp * Bu_exp).sum(dim=-2)  # (..., m, n)
    L = -U

    # Expand symbolic parts to (..., m, k, n).
    As_exp = As.unsqueeze(-1).expand(*As.shape, n)
    Bs_exp = Bs.unsqueeze(-3).expand(*Bs.shape[:-2], m, k, n)

    a = torch.sqrt(Au_exp)
    b = torch.sqrt(Bu_exp)
    lama = torch.nan_to_num(a / b, nan=1.0, posinf=1.0)
    lamb = torch.nan_to_num(b / a, nan=1.0, posinf=1.0)

    ep = lama * As_exp + lamb * Bs_exp
    em = lama * As_exp - lamb * Bs_exp

    # ub(x²/4) = max(|lb(x)|, |ub(x)|)²/4  (needed since ep/em may not be symmetric)
    Pos = torch.nan_to_num(
        torch.maximum(ep.ub().abs(), ep.lb().abs()) ** 2 / 4,
        nan=1e10, posinf=1e10, neginf=1e10,
    )
    Neg = torch.nan_to_num(
        -torch.maximum(em.ub().abs(), em.lb().abs()) ** 2 / 4,
        nan=-1e10, posinf=-1e10, neginf=-1e10,
    )

    U = torch.minimum(Pos.sum(dim=-2), U)
    L = torch.maximum(Neg.sum(dim=-2), L)

    result += (U + L) / 2 + (U - L) / 2 * LpEpsilon(result.shape)
    return result


def bilinear_matmul(A: Expr, B: Expr) -> Expr:
    """Linearize ``A @ B`` when both operands are symbolic expressions."""
    return square_matmul(A, B)


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
