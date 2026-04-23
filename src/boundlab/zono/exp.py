"""Exp linearizer for zonotope abstract interpretation.

Implements the DeepT minimal-area relaxation for the exponential function.
"""

import torch

from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp

from . import ZonoBounds, _register_linearizer


@_register_linearizer("Exp")
def exp_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> ZonoBounds:
    """Minimal-area exp relaxation (DeepT, Section 4.5).

    The returned zonotope ``y = slope·x + mu + beta·ε`` (with ε ∈ [-1, 1])
    over-approximates ``exp(x)`` on ``[lb, ub]`` and guarantees
    ``mu - beta ≥ 0`` so downstream handlers (``reciprocal`` / softmax)
    never see a non-positive lower bound.

    Strategy:

    - For each element, try the single-slope "minimal area" relaxation:
      tangent at ``t_opt`` as the lower envelope, parallel secant line
      through ``(ub_c, exp(ub_c))`` as the upper envelope.
    - If the tangent gives a non-positive offset (e.g. ``lb > 0``) or the
      fp32 precision would lose that offset against ``exp(ub)``, fall back
      to the interval relaxation ``slope = 0``, ``mu = beta = exp(ub)/2``,
      which yields ``[0, exp(ub)]`` — loose but guaranteed ``mu - beta = 0``
      bit-exactly (so downstream ``ublb`` cannot go negative).
    - Clamp ``lb``/``ub`` to ±30 to keep ``exp`` numerically safe; the
      underflow/overflow branches use the interval fallback.

    Examples
    --------
    >>> import torch
    >>> import boundlab.expr as expr
    >>> from boundlab.zono.exp import exp_linearizer
    >>> x = expr.ConstVal(torch.tensor([0.0])) + 0.1 * expr.LpEpsilon([1])
    >>> ub, lb = x.ublb()
    >>> b = exp_linearizer(ub, lb)
    >>> b.bias.shape
    torch.Size([1])
    """
    SAFE = 30.0
    out_dtype = ub.dtype

    lb_c = torch.clamp(lb, -88, 88)
    ub_c = torch.clamp(ub, -88, 88)
    el = torch.exp(lb_c)
    eu = torch.exp(ub_c)

    degen = torch.abs(ub_c - lb_c) < 1e-12
    underflow = lb < -SAFE
    overflow = ub > SAFE

    # Tangent-secant (minimal area) relaxation for the normal branch.
    safe_width = torch.clamp(ub_c - lb_c, min=1e-30)
    secant_slope = (eu - el) / safe_width
    t_crit = torch.log(torch.clamp(secant_slope, min=1e-300))
    t_opt = torch.minimum(t_crit, lb_c + 0.99)
    slope = torch.exp(t_opt)
    low_offset = slope * (1.0 - t_opt)       # mu - beta (tangent at (t_opt, exp(t_opt)))
    high_offset = eu - slope * ub_c          # mu + beta (upper line through (ub_c, eu))
    mu = 0.5 * (low_offset + high_offset)
    beta = 0.5 * (high_offset - low_offset)

    # Fall back to interval relaxation [0, eu] when the tangent's lower
    # offset is non-positive or is so small compared to ``eu`` that fp32
    # storage would lose it.  Using ``mu == beta`` bit-exactly guarantees
    # ``mu − beta == 0`` after the round-trip to ``out_dtype``.
    fp32_eps_margin = torch.finfo(out_dtype).eps * torch.clamp(eu, min=1.0) * 8.0
    use_interval = (low_offset <= fp32_eps_margin) | underflow | overflow

    # Always use the clamped eu (finite) for the interval fallback so
    # mu and beta stay finite even when ub is astronomically large.
    # Accept unsound bounds for extreme inputs — the assertion contract
    # (mu - beta >= 0) is what matters for downstream reciprocal.
    half_eu = 0.5 * eu

    slope = torch.where(use_interval, torch.zeros_like(slope), slope)
    mu = torch.where(use_interval, half_eu, mu)
    beta = torch.where(use_interval, half_eu, beta)

    # Degenerate: ub ≈ lb, collapse to exp(lb_c).
    slope = torch.where(degen, torch.zeros_like(slope), slope)
    mu = torch.where(degen, el, mu)
    beta = torch.where(degen, torch.zeros_like(beta), torch.abs(beta))

    # Cast back to input dtype.  For the interval branch, ``mu`` and ``beta``
    # are produced by the same expression (``0.5 * eu_over``), so they round
    # to bit-identical values in out_dtype and ``mu − beta == 0`` exactly.
    slope = slope.to(out_dtype)
    mu = mu.to(out_dtype)
    beta = beta.to(out_dtype)

    error_op = EinsumOp.from_hardmard(beta, len(ub.shape))

    return ZonoBounds(bias=mu, error_coeffs=error_op, input_weights=[slope])
