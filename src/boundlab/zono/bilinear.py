"""Bilinear operation handlers for zonotope abstract interpretation.

Provides McCormick-style linearization for matmul and element-wise product
when both operands are symbolic expressions (Expr @ Expr or Expr * Expr).
"""

import torch

from boundlab.expr._core import Expr
from boundlab.expr._affine import ConstVal
from boundlab.expr._var import LpEpsilon


def bilinear_matmul(A: Expr, B: Expr) -> Expr:
    r"""Linearize ``A @ B`` when both operands are symbolic expressions.

    A: (m, k), B: (k, n) → result: (m, n)

    The method uses a first-order expansion around expression centers:

    .. math::

       A B \approx c_A B + A c_B - c_A c_B + E

    where :math:`E` is bounded by interval half-widths:

    .. math::

       |E| \le \mathrm{hw}(A)\,\mathrm{hw}(B)

    and represented using fresh noise symbols.

    Args:
        A: Left expression with shape ``(m, k)``.
        B: Right expression with shape ``(k, n)``.

    Returns:
        An expression over-approximating ``A @ B``.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.bilinear import bilinear_matmul
    >>> A = expr.ConstVal(torch.ones(2, 3)) + 0.1 * expr.LpEpsilon([2, 3])
    >>> B = expr.ConstVal(torch.ones(3, 4)) + 0.1 * expr.LpEpsilon([3, 4])
    >>> C = bilinear_matmul(A, B)
    >>> C.shape
    torch.Size([2, 4])
    """
    assert len(A.shape) >= 2 and len(B.shape) >= 2, \
        f"Need at least 2D for matmul, got {A.shape} @ {B.shape}"
    assert A.shape[-1] == B.shape[-2], \
        f"Inner dims must match: {A.shape} @ {B.shape}"

    Ac, As = A.symmetric_decompose()  # Ac: constant part, As: epsilon part
    Bc, Bs = B.symmetric_decompose()  # Bc: constant part, Bs: epsilon part

    result = Ac @ Bs + As @ Bc + Ac @ Bc

    # Error bound: |E| ≤ hw(A) * hw(B) where hw = half-width
    if As.is_symmetric_to_0():
        Ahw = As.ub()
    else:
        A_ub, A_lb = As.ublb()
        Ahw = (A_ub - A_lb) / 2.0

    if Bs.is_symmetric_to_0():
        error_bound = (Ahw @ Bs).ub()
    else:
        B_ub, B_lb = Bs.ublb()
        Bhw = (B_ub - B_lb) / 2.0
        error_bound = Ahw @ Bhw

    new_eps = LpEpsilon(error_bound.shape)
    result = result + error_bound * new_eps

    return result


def bilinear_elementwise(A: Expr, B: Expr) -> Expr:
    r"""Linearize element-wise product of two symbolic expressions.

    Both A and B must have the same shape.

    The approximation is:

    .. math::

       A \odot B \approx c_A \odot B + A \odot c_B - c_A \odot c_B + E

    with element-wise error bound:

    .. math::

       |E| \le \mathrm{hw}(A) \odot \mathrm{hw}(B)

    Args:
        A: First expression.
        B: Second expression (same shape as ``A``).

    Returns:
        An expression over-approximating ``A * B``.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.bilinear import bilinear_elementwise
    >>> A = expr.ConstVal(torch.ones(3)) + 0.2 * expr.LpEpsilon([3])
    >>> B = expr.ConstVal(torch.zeros(3)) + 0.3 * expr.LpEpsilon([3])
    >>> C = bilinear_elementwise(A, B)
    >>> C.shape
    torch.Size([3])
    """
    assert A.shape == B.shape, \
        f"Shapes must match for element-wise product: {A.shape} vs {B.shape}"

    Ac, As = A.symmetric_decompose()  # Ac: constant part, As: zero-constant part
    Bc, Bs = B.symmetric_decompose()  # Bc: constant part, Bs: zero-constant part

    result = Ac * Bs + As * Bc + Ac * Bc

    # Error bound: |E| ≤ hw(A) * hw(B)
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
    result = result + error_bound * new_eps

    return result


def matmul_handler(A, B):
    """Dispatcher implementation for ``torch.matmul``.

    Routing rules:

    - ``Expr @ Expr``: McCormick-style bilinear relaxation.
    - ``Expr @ Tensor`` or ``Tensor @ Expr``: exact affine path.
    - ``Tensor @ Tensor``: delegated to ``torch.matmul``.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.bilinear import matmul_handler
    >>> A = expr.ConstVal(torch.ones(1, 2)) + expr.LpEpsilon([1, 2])
    >>> B = torch.ones(2, 1)
    >>> matmul_handler(A, B).shape
    torch.Size([1, 1])
    """
    if isinstance(A, Expr) and isinstance(B, Expr):
        return bilinear_matmul(A, B)
    elif isinstance(A, Expr) and isinstance(B, torch.Tensor):
        return A @ B  # Expr.__matmul__(Tensor)
    elif isinstance(A, torch.Tensor) and isinstance(B, Expr):
        return A @ B  # Tensor.__matmul__ → Expr.__rmatmul__(Tensor)
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return torch.matmul(A, B)
    else:
        return A @ B
