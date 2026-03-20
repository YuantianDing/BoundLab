"""Bilinear operation handlers for zonotope abstract interpretation.

Provides McCormick-style linearization for matmul and element-wise product
when both operands are symbolic expressions (Expr @ Expr or Expr * Expr).
"""

import torch

from boundlab.expr._core import Expr
from boundlab.expr._affine import ConstVal
from boundlab.expr._var import LpEpsilon


def bilinear_matmul(A: Expr, B: Expr) -> Expr:
    """Compute A @ B where both are Expr using McCormick linearization.

    A: (m, k), B: (k, n) → result: (m, n)

    Linearization: c_A @ B + A @ c_B - c_A @ c_B + error
    Error bound:   hw_A @ hw_B  (half-widths of A and B)
    """
    assert len(A.shape) == 2 and len(B.shape) == 2, \
        f"Only 2D matmul supported, got {A.shape} @ {B.shape}"
    assert A.shape[1] == B.shape[0], \
        f"Inner dims must match: {A.shape} @ {B.shape}"

    a_c = A.center()  # (m, k) concrete tensor
    b_c = B.center()  # (k, n) concrete tensor

    # Affine linearization around centers
    term1 = a_c @ B                     # c_A @ B (linear in B, uses Expr.__rmatmul__)
    term2 = A @ b_c                     # A @ c_B (linear in A, uses Expr.__matmul__)
    const = ConstVal(-(a_c @ b_c))      # -c_A @ c_B (constant)

    result = term1 + term2 + const

    # Error bound for quadratic remainder: |δA @ δB| ≤ hw_A @ hw_B
    ub_A, lb_A = A.ublb()
    ub_B, lb_B = B.ublb()
    hw_a = (ub_A - lb_A) / 2  # (m, k)
    hw_b = (ub_B - lb_B) / 2  # (k, n)
    error_bound = hw_a @ hw_b  # (m, n) concrete tensor

    # Introduce new error symbols
    new_eps = LpEpsilon(error_bound.shape)
    result = result + error_bound * new_eps

    return result


def bilinear_elementwise(A: Expr, B: Expr) -> Expr:
    """Element-wise product of two Exprs using McCormick linearization.

    Both A and B must have the same shape.

    Linearization: c_A * B + A * c_B - c_A * c_B + error
    Error bound:   hw_A * hw_B  (element-wise half-widths)
    """
    assert A.shape == B.shape, \
        f"Shapes must match for element-wise product: {A.shape} vs {B.shape}"

    a_c = A.center()  # concrete tensor
    b_c = B.center()  # concrete tensor

    # Affine linearization
    # a_c * B: Tensor * Expr → Expr.__rmul__(Tensor)
    # A * b_c: Expr.__mul__(Tensor)
    result = a_c * B + A * b_c + ConstVal(-(a_c * b_c))

    # Error bound
    ub_A, lb_A = A.ublb()
    ub_B, lb_B = B.ublb()
    hw_a = (ub_A - lb_A) / 2
    hw_b = (ub_B - lb_B) / 2
    error_bound = hw_a * hw_b  # element-wise

    new_eps = LpEpsilon(error_bound.shape)
    result = result + error_bound * new_eps

    return result


def matmul_handler(A, B):
    """Dispatcher handler for torch.matmul.

    Routes to the appropriate implementation based on operand types.
    """
    if isinstance(A, Expr) and isinstance(B, Expr):
        return bilinear_matmul(A, B)
    elif isinstance(A, Expr) and isinstance(B, torch.Tensor):
        return A @ B  # Expr.__matmul__(Tensor)
    elif isinstance(A, torch.Tensor) and isinstance(B, Expr):
        return A @ B  # Tensor.__matmul__ → Expr.__rmatmul__(Tensor)
    else:
        return torch.matmul(A, B)
