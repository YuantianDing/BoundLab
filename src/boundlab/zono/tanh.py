"""Tanh linearizer for zonotope abstract interpretation.

Implements the DeepT minimal-area relaxation for hyperbolic tangent.
"""

import torch

from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp
# from .softmax2 import softmax2_lb, softmax2_ub2

from . import ZonoBounds, _register_linearizer

@_register_linearizer("tanh")
def tanh_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> ZonoBounds:
    """Minimal-area tanh relaxation (DeepT, Section 4.4).

    y = lambda*x + mu + beta*eps_new
    lambda = min(sech^2(l), sech^2(u)) = min(1-tanh^2(l), 1-tanh^2(u))
    mu = 0.5*(tanh(u) + tanh(l) - lambda*(u + l))
    beta = 0.5*(tanh(u) - tanh(l) - lambda*(u - l))

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.tanh import tanh_linearizer
    >>> x = expr.ConstVal(torch.tensor([0.0])) + expr.LpEpsilon([1])
    >>> ub, lb = x.ublb()
    >>> b = tanh_linearizer(ub, lb)
    >>> b.bias.shape
    torch.Size([1])
    """
    output_shape = ub.shape

    degen = torch.abs(ub - lb) < 1e-12

    tl = torch.tanh(lb)
    tu = torch.tanh(ub)

    slope = torch.minimum(1 - tl**2, 1 - tu**2)
    mu = 0.5 * (tu + tl - slope * (ub + lb))
    beta = 0.5 * (tu - tl - slope * (ub - lb))

    slope = torch.where(degen, 1 - torch.tanh(lb)**2, slope)
    mu = torch.where(degen, torch.zeros_like(mu), mu)
    beta = torch.where(degen, torch.zeros_like(beta), torch.abs(beta))

    # Build ZonoBounds
    error_op = EinsumOp.from_hardmard(beta, len(ub.shape))

    return ZonoBounds(bias=mu, error_coeffs=error_op, input_weights=[slope])


# def tanh_linearizer2(ub: torch.Tensor, lb: torch.Tensor) -> ZonoBounds:
#     """Tanh linearizer implemented via softmax2 bounds.

#     Uses ``tanh(x) = 2 * softmax2(1, -2x) - 1`` with shared-slope affine bounds
#     from ``softmax2_ub2`` / ``softmax2_lb``.
#     """
#     # print((ub - lb).mean().item(), (ub - lb).std().item())
#     # print((ub + lb).mean().item() / 2, ((ub + lb) / 2).std().item())
#     degen = torch.abs(ub - lb) < 1e-12
#     # softmax2 helper parameter (lam_y) range is [-1, 0].
#     # For tanh: slope_tanh = -4 * lam_y, so lam_y in [-0.25, 0].
#     # Use the DeepT minimal-area slope choice: min(sech^2(lb), sech^2(ub)).
#     slope_tanh = torch.minimum(1 - torch.tanh(lb) ** 2, 1 - torch.tanh(ub) ** 2)
#     # slope_tanh = (torch.tanh(ub) - torch.tanh(lb)) / (ub - lb + 1e-30)
#     lam_y = -slope_tanh / 4
#     lam_y = torch.clamp(lam_y, min=-1.0, max=-1e-8)

#     # y = -2x maps x in [lb, ub] to y in [-2ub, -2lb].
#     y_lb = -2 * ub
#     y_ub = -2 * lb
#     # x_one = torch.ones_like(ub)

#     # Bounds for s(x) = softmax2(1, -2x):
#     #   s <= m_s * x + b_s_ub, s >= m_s * x + b_s_lb, m_s = -2*lam_y.
#     b_s_ub = softmax2_ub2(lam_y, 1, y_ub, y_lb)
#     b_s_lb = softmax2_lb(lam_y, 1, y_ub, y_lb)
#     m_s = -2 * lam_y

#     # tanh(x) = 2*s(x) - 1
#     slope = 2 * m_s
#     b_t_ub = 2 * b_s_ub - 1
#     b_t_lb = 2 * b_s_lb - 1
#     mu = (b_t_ub + b_t_lb) / 2
#     beta = ((b_t_ub - b_t_lb) / 2).abs()

#     # Exact singleton fallback.
#     slope = torch.where(degen, 1 - torch.tanh(lb) ** 2, slope)
#     mu = torch.where(degen, torch.tanh(lb) - slope * lb, mu)
#     beta = torch.where(degen, torch.zeros_like(beta), beta)

#     # Build ZonoBounds
#     error_op = EinsumOp.from_hardmard(beta, len(ub.shape))

#     return ZonoBounds(bias=mu, error_coeffs=error_op, input_weights=[slope])
