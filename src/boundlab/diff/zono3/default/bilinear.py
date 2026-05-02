"""

Scalar Multiplication (element-wise):
    a1·b1 − a2·b2 = a1·∆b + ∆a·b2

Dot Product / Matrix Multiply:
    A1@B1 − A2@B2 = A1@∆B + ∆A@B2

"""

import torch

from boundlab.expr._core import Expr
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.zono.bilinear import (
    bilinear_elementwise,
    bilinear_matmul,
)


def diff_bilinear_elementwise(a: DiffExpr3, b: DiffExpr3) -> DiffExpr3:
    assert a.shape == b.shape, \
        f"Shapes must match: {a.shape} vs {b.shape}"

    out_x = bilinear_elementwise(a.x, b.x)
    out_y = bilinear_elementwise(a.y, b.y)

    # Diff: a1·Δb + Δa·b2
    term1 = bilinear_elementwise(a.x, b.diff)
    term2 = bilinear_elementwise(a.diff, b.y)
    out_diff = term1 + term2

    return DiffExpr3(out_x, out_y, out_diff)


def diff_bilinear_matmul(a: DiffExpr3, b: DiffExpr3) -> DiffExpr3:

    out_x = bilinear_matmul(a.x, b.x)
    out_y = bilinear_matmul(a.y, b.y)

    # Diff: A1@(B1 - B2) + (A1 - A2)@B2
    term1 = bilinear_matmul(a.x, b.diff)
    term2 = bilinear_matmul(a.diff, b.y)
    out_diff = term1 + term2

    # Reset: if Z_Δ bound is wider than Z_x - Z_y, swap per-neuron.
    from boundlab.prop import bound_width
    sub_diff = out_x - out_y
    bw_d = bound_width(out_diff)
    bw_s = bound_width(sub_diff)
    n_reset = (bw_s < bw_d).sum().item()
    n_total = bw_d.numel()
    max_d = bw_d.max().item()
    max_s = bw_s.max().item()
    print(f"  [matmul reset] {n_reset}/{n_total} neurons reset, "
          f"max bw_d={max_d:.3e}, max bw_s={max_s:.3e}")
    mask = (bw_s < bw_d).float()
    out_diff = mask * sub_diff + (1.0 - mask) * out_diff

    return DiffExpr3(out_x, out_y, out_diff)


def diff_mul_handler(a, b):
    if isinstance(a, DiffExpr3) and isinstance(b, DiffExpr3):
        return diff_bilinear_elementwise(a, b)

    if isinstance(a, DiffExpr3) and isinstance(b, DiffExpr2):
        try:
            return a * b
        except TypeError:
            b3 = DiffExpr3(b.x, b.y, b.x - b.y)
            return diff_bilinear_elementwise(a, b3)

    if isinstance(a, DiffExpr2) and isinstance(b, DiffExpr3):
        try:
            return b * a
        except TypeError:
            a3 = DiffExpr3(a.x, a.y, a.x - a.y)
            return diff_bilinear_elementwise(a3, b)

    if isinstance(a, DiffExpr2) and isinstance(b, DiffExpr2):
        try:
            return a * b
        except TypeError:
            a3 = DiffExpr3(a.x, a.y, a.x - a.y)
            b3 = DiffExpr3(b.x, b.y, b.x - b.y)
            return diff_bilinear_elementwise(a3, b3)

    if isinstance(a, (DiffExpr3, DiffExpr2)):
        return a * b
    if isinstance(b, (DiffExpr3, DiffExpr2)):
        return b * a

    if isinstance(a, Expr) and isinstance(b, torch.Tensor):
        return a * b
    if isinstance(a, torch.Tensor) and isinstance(b, Expr):
        return b * a
    return a * b


def diff_matmul_handler(a, b):
    from boundlab.zono.bilinear import matmul_handler as std_matmul_handler

    if isinstance(a, DiffExpr3) and isinstance(b, DiffExpr3):
        return diff_bilinear_matmul(a, b)

    if isinstance(a, DiffExpr3) and isinstance(b, DiffExpr2):
        if b.is_constant():
            return a @ b
        b3 = DiffExpr3(b.x, b.y, b.x - b.y)
        return diff_bilinear_matmul(a, b3)

    if isinstance(a, DiffExpr2) and isinstance(b, DiffExpr3):
        return a @ b

    if isinstance(a, DiffExpr2) and isinstance(b, DiffExpr2):
        if a.is_constant() or b.is_constant():
            return a @ b
        a3 = DiffExpr3(a.x, a.y, a.x - a.y)
        b3 = DiffExpr3(b.x, b.y, b.x - b.y)
        return diff_bilinear_matmul(a3, b3)

    if isinstance(a, (DiffExpr3, DiffExpr2)):
        return a._map_all(lambda x: std_matmul_handler(x, b))
    if isinstance(b, (DiffExpr3, DiffExpr2)):
        return b._map_all(lambda x: std_matmul_handler(a, x))

    return std_matmul_handler(a, b)


__all__ = [
    "diff_bilinear_elementwise",
    "diff_bilinear_matmul",
    "diff_mul_handler",
    "diff_matmul_handler",
]