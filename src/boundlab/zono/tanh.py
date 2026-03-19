"""Tanh linearizer for zonotope abstract interpretation.

Implements the DeepT minimal-area relaxation for hyperbolic tangent.
"""

from ast import expr

import torch

from boundlab.expr._core import Expr
from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp

from . import ZonoBounds, _register_linearizer


@_register_linearizer("tanh")
def tanh_linearizer(expr: Expr) -> ZonoBounds:
    """Minimal-area tanh relaxation (DeepT, Section 4.4).

    y = lambda*x + mu + beta*eps_new
    lambda = min(sech^2(l), sech^2(u)) = min(1-tanh^2(l), 1-tanh^2(u))
    mu = 0.5*(tanh(u) + tanh(l) - lambda*(u + l))
    beta = 0.5*(tanh(u) - tanh(l) - lambda*(u - l))
    """
    lb = expr.lb()
    ub = expr.ub()
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
    error_op = EinsumOp.from_hardmard(beta, len(expr.shape))

    return ZonoBounds(bias=mu, error_coeffs=error_op, input_weights=[slope])
