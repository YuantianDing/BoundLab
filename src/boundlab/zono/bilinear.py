"""Bilinear operation handlers for zonotope abstract interpretation.

Provides McCormick-style linearization for matmul and element-wise product
when both operands are symbolic expressions (Expr @ Expr or Expr * Expr),
plus the tighter DeepT-Precise relaxation from Bonaert et al. (2021).
"""

from functools import reduce
import operator
from typing import Union

import torch

from boundlab.expr._core import Expr
from boundlab.expr._affine import AffineSum, ConstVal
from boundlab.expr._var import LpEpsilon
from boundlab.linearop._base import LinearOp


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

    Ac, As = A.split_const()  # Ac: constant part, As: epsilon part
    Bc, Bs = B.split_const()  # Bc: constant part, Bs: epsilon part

    result = Ac @ Bs + As @ Bc + Ac @ Bc

    # Error bound: |E| ≤ hw(A) * hw(B) where hw = half-width
    assert As.is_symmetric_to_0() and Bs.is_symmetric_to_0()
    
    error_bound = As.ub() @ Bs.ub()


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

    Ac, As = A.split_const()  # Ac: constant part, As: zero-constant part
    Bc, Bs = B.split_const()  # Bc: constant part, Bs: zero-constant part

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


def _eps_children(e: Expr) -> dict[LpEpsilon, LinearOp]:
    """Return ``{LpEpsilon: LinearOp}`` for the symbolic part of ``e``.

    ``e`` is assumed to be the zero-constant half of a symmetric decomposition:
    either an :class:`AffineSum` whose children are :class:`LpEpsilon` nodes,
    or a :class:`ConstVal` (empty children).
    """
    if isinstance(e, AffineSum):
        children = dict(e.children_dict)
        for child in children:
            assert isinstance(child, LpEpsilon), \
                f"DeepT-Precise requires LpEpsilon children, got {type(child).__name__}"
        return children
    return {}

def deept_precise_matmul(A: Expr, B: Expr) -> Expr:
    r"""DeepT-Precise relaxation of ``A @ B`` (2D operands only).

    The bilinear error :math:`A_s B_s` is expanded by enumerating every pair
    of noise symbols ``(eps_A, eps_B)`` with ``eps_A`` drawn from ``A`` and
    ``eps_B`` from ``B``. For each pair:

    1. Materialise both jacobians.
    2. Contract the ``k`` dim per matmul rules to get
       :math:`T[i, j, s_A, s_B] = \sum_l \alpha_{eps_A, s_A}[i, l]\,
                                         \beta_{eps_B, s_B}[l, j]`.
    3. If ``eps_A is eps_B`` (shared epsilon), the positions where
       ``s_A == s_B`` represent :math:`\epsilon^2 \in [0, 1]`: they contribute
       ``relu(T)`` to the upper-error tensor and ``relu(-T)`` to the
       lower-error tensor. Everywhere else (and every position of mismatched
       pairs) corresponds to :math:`\epsilon_i \epsilon_j \in [-1, 1]` and
       contributes ``|T|`` to both.
    4. Sum over all input dims and accumulate into ``upper_err`` / ``lower_err``.

    The resulting asymmetric bilinear interval ``[-lower_err, +upper_err]``
    is repackaged as ``center + half_width · ε_new`` where
    ``center = (upper_err - lower_err) / 2`` and
    ``half_width = (upper_err + lower_err) / 2``.

    Args:
        A: Left operand with shape ``(m, k)``.
        B: Right operand with shape ``(k, n)``.

    Returns:
        An expression over-approximating ``A @ B``.
    """
    assert len(A.shape) >= 2 and len(B.shape) >= 2, \
        f"Need at least 2D for matmul, got {A.shape} @ {B.shape}"
    assert A.shape[:-2] == B.shape[:-2], \
        f"Batch dims must match: {A.shape} @ {B.shape}"
    assert A.shape[-1] == B.shape[-2], \
        f"Inner dims must match: {A.shape} @ {B.shape}"

    Ac, As = A.split_const()
    Bc, Bs = B.split_const()

    result = Ac @ Bs + As @ Bc + Ac @ Bc

    A_children = _eps_children(As)
    B_children = _eps_children(Bs)
    
    b = reduce(operator.mul, A.shape[:-2], 1)
    m, k, n = A.shape[-2], A.shape[-1], B.shape[-1]

    err = torch.zeros(b, m, n)
    for eps_A, op_A in A_children.items():
        jac_A = op_A.jacobian().reshape(b, m, k, -1)
        for eps_B, op_B in B_children.items():
            jac_B = op_B.jacobian().reshape(b, k, n, -1)
            T = torch.einsum("bmki, bknj->bmnij", jac_A, jac_B)
            err += T.abs().sum(dim=(-2, -1))

    err = err.reshape(A.shape[:-2] + (m, n))
    new_eps = LpEpsilon(err.shape)
    result = result + err * new_eps
    return result

def square_matmul(A: Expr, B: Expr) -> Expr:
    assert len(A.shape) >= 2 and len(B.shape) >= 2, \
        f"Need at least 2D for matmul, got {A.shape} @ {B.shape}"
    assert A.shape[:-2] == B.shape[:-2], \
        f"Batch dims must match: {A.shape} @ {B.shape}"
    assert A.shape[-1] == B.shape[-2], \
        f"Inner dims must match: {A.shape} @ {B.shape}"

    Ac, As = A.split_const()
    Bc, Bs = B.split_const()

    result = Ac @ Bs + As @ Bc + Ac @ Bc
    assert As.is_symmetric_to_0() and Bs.is_symmetric_to_0()
    
    U = (As + Bs).ub() ** 2
    L = -(As - Bs).ub() ** 2

    result += (U + L) / 2 + (U - L) / 2 * LpEpsilon(U.shape)
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
    if isinstance(A, Expr) and _is_const(B):
        return A @ B  # Expr.__matmul__(Tensor)
    elif _is_const(A) and isinstance(B, Expr):
        return A @ B  # Tensor.__matmul__ → Expr.__rmatmul__(Tensor)
    elif _is_const(A) and _is_const(B):
        return torch.matmul(A, B)
    elif isinstance(A, Expr) and isinstance(B, Expr):
        # precise = deept_precise_matmul(A, B)
        normal = bilinear_matmul(A, B)
        return normal
    else:
        return A @ B

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


def _is_const(tensor: Union[torch.Tensor, Expr, int, float]) -> bool:
    return isinstance(tensor, torch.Tensor) or isinstance(tensor, ConstVal) or isinstance(tensor, (int, float))
