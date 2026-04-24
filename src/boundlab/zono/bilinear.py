"""Bilinear operation handlers for zonotope abstract interpretation.

Provides McCormick-style linearization for matmul and element-wise product
when both operands are symbolic expressions (Expr @ Expr or Expr * Expr),
plus the tighter DeepT-Precise relaxation from Bonaert et al. (2021).
"""

from typing import Union

import torch

from boundlab.expr._core import Expr
from boundlab.expr._affine import AffineSum, ConstVal
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


def _eps_children(e: Expr) -> dict:
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


def _half_width(e: Expr) -> torch.Tensor:
    """Half-width of a symmetric (zero-centered) expression."""
    if e.is_symmetric_to_0():
        return e.ub()
    ub, lb = e.ublb()
    return (ub - lb) / 2.0


def deept_precise_elementwise(A: Expr, B: Expr) -> Expr:
    r"""DeepT-Precise relaxation of element-wise ``A * B``.

    Given :math:`A = c_A + \sum_i \alpha_i \epsilon_i` and
    :math:`B = c_B + \sum_j \beta_j \epsilon_j`, expand the product and
    bound every :math:`\epsilon_i^2` term by :math:`[0, 1]` and every
    :math:`\epsilon_i \epsilon_j` (:math:`i \ne j`) by :math:`[-1, 1]`.
    Correlations between distinct cross-product terms are ignored (they are
    summed as independent intervals), following Bonaert et al. (2021).

    The resulting approximation error is bounded by a single fresh noise
    symbol with half-width

    .. math::

       \mathrm{hw} = \mathrm{hw}(A) \odot \mathrm{hw}(B)
            \;-\; \tfrac{1}{2} \sum_{\epsilon \in A \cap B, s}
                  |\alpha_{\epsilon, s}\,\beta_{\epsilon, s}|

    and a constant center shift
    :math:`\tfrac{1}{2} \sum_{\epsilon, s} \alpha_{\epsilon, s}\,\beta_{\epsilon, s}`
    coming from the :math:`\epsilon_i^2 \in [0, 1]` diagonal terms. Compared
    with :func:`bilinear_elementwise`, the diagonal contribution is resolved
    exactly instead of being over-approximated by the symmetric interval
    :math:`[-|\alpha\beta|, |\alpha\beta|]`.

    Args:
        A: First operand; expected to be an :class:`AffineSum` whose symbolic
            children are :class:`LpEpsilon` instances.
        B: Second operand (same shape as ``A``), same assumption as ``A``.

    Returns:
        An expression over-approximating ``A * B``.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.bilinear import deept_precise_elementwise
    >>> eps = expr.LpEpsilon([3])
    >>> A = expr.ConstVal(torch.ones(3)) + 0.2 * eps
    >>> B = expr.ConstVal(torch.zeros(3)) + 0.3 * eps
    >>> C = deept_precise_elementwise(A, B)
    >>> C.shape
    torch.Size([3])
    """
    assert A.shape == B.shape, \
        f"Shapes must match for element-wise product: {A.shape} vs {B.shape}"

    Ac, As = A.symmetric_decompose()
    Bc, Bs = B.symmetric_decompose()

    # Affine baseline: A*B = Ac*Bc + Ac*Bs + As*Bc + (As*Bs bilinear part).
    result = Ac * Bs + As * Bc + Ac * Bc

    A_children = _eps_children(As)
    B_children = _eps_children(Bs)

    diag_center = torch.zeros(A.shape)
    diag_abs = torch.zeros(A.shape)
    for eps in set(A_children) & set(B_children):
        jac_A = A_children[eps].jacobian()  # shape: A.shape + eps.shape
        jac_B = B_children[eps].jacobian()  # shape: B.shape + eps.shape
        product = jac_A * jac_B
        eps_ndim = len(eps.shape)
        if eps_ndim:
            reduce_dims = tuple(range(-eps_ndim, 0))
            diag_center = diag_center + product.sum(dim=reduce_dims)
            diag_abs = diag_abs + product.abs().sum(dim=reduce_dims)
        else:
            diag_center = diag_center + product
            diag_abs = diag_abs + product.abs()

    A_hw = _half_width(As)
    B_hw = _half_width(Bs)

    # Diag interval [0, αβ] or [αβ, 0] per ε_i² → center αβ/2, half-width |αβ|/2.
    # Off-diag interval [-|αβ'|, |αβ'|] per ε_i ε_j (i≠j) → sum is
    # A_hw·B_hw − diag_abs. Combined total half-width absorbs both into one
    # fresh noise symbol.
    center = diag_center / 2.0
    total_hw = (A_hw * B_hw - diag_abs / 2.0).clamp(min=0.0)

    result = result + center
    new_eps = LpEpsilon(total_hw.shape)
    result = result + total_hw * new_eps
    return result


def deept_precise_matmul(A: Expr, B: Expr) -> Expr:
    r"""DeepT-Precise relaxation of the 2D matmul ``A @ B``.

    Extends :func:`deept_precise_elementwise` across the matmul contraction.
    For every shared noise symbol :math:`\epsilon` and index :math:`s`,
    the diagonal coefficient is

    .. math::

       D_{\epsilon, s}[i, j] = \sum_l \alpha^A_{\epsilon, s}[i, l]\,
                                     \beta^B_{\epsilon, s}[l, j],

    whose :math:`\epsilon_s^2 \in [0, 1]` interval yields a center offset
    :math:`D/2` and a half-width :math:`|D|/2`. The remaining off-diagonal
    half-width is :math:`A_{\mathrm{hw}} \mathbin{@} B_{\mathrm{hw}}` minus
    the per-:math:`(\epsilon, s, l)` diagonal absolute contribution
    :math:`\sum_l |\alpha^A_{\epsilon, s}[i, l]\,\beta^B_{\epsilon, s}[l, j]|`.

    Args:
        A: Left 2D operand with shape ``(m, k)``.
        B: Right 2D operand with shape ``(k, n)``.

    Returns:
        An expression over-approximating ``A @ B``.
    """
    assert len(A.shape) == 2 and len(B.shape) == 2, \
        f"deept_precise_matmul only supports 2D inputs, got {A.shape} @ {B.shape}"
    assert A.shape[-1] == B.shape[-2], \
        f"Inner dims must match: {A.shape} @ {B.shape}"

    Ac, As = A.symmetric_decompose()
    Bc, Bs = B.symmetric_decompose()

    result = Ac @ Bs + As @ Bc + Ac @ Bc

    A_children = _eps_children(As)
    B_children = _eps_children(Bs)

    m, k = A.shape
    n = B.shape[1]
    out_shape = torch.Size([m, n])

    diag_center = torch.zeros(out_shape)
    diag_hw = torch.zeros(out_shape)
    diag_pointwise_abs = torch.zeros(out_shape)
    for eps in set(A_children) & set(B_children):
        jac_A = A_children[eps].jacobian()  # (m, k, *S)
        jac_B = B_children[eps].jacobian()  # (k, n, *S)
        jac_A_flat = jac_A.reshape(m, k, -1)
        jac_B_flat = jac_B.reshape(k, n, -1)

        # D[i, j, s] = Σ_l α_A[i, l, s] β_B[l, j, s]
        D = torch.einsum("iks,kjs->ijs", jac_A_flat, jac_B_flat)
        diag_center = diag_center + D.sum(dim=-1)
        diag_hw = diag_hw + D.abs().sum(dim=-1)
        # Per-(l, s) absolute diagonal contribution for off-diag correction.
        diag_pointwise_abs = diag_pointwise_abs + torch.einsum(
            "iks,kjs->ij", jac_A_flat.abs(), jac_B_flat.abs()
        )

    A_hw = _half_width(As)
    B_hw = _half_width(Bs)

    center = diag_center / 2.0
    total_hw = (diag_hw / 2.0 + A_hw @ B_hw - diag_pointwise_abs).clamp(min=0.0)

    result = result + center
    new_eps = LpEpsilon(out_shape)
    result = result + total_hw * new_eps
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
        return bilinear_matmul(A, B)
    else:
        return A @ B

def _is_const(tensor: Union[torch.Tensor, Expr]) -> bool:
    return isinstance(tensor, torch.Tensor) or isinstance(tensor, ConstVal)