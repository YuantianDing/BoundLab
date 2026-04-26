"""Softmax2 handler for zonotope abstract interpretation.

Defines

.. math::

   \mathrm{softmax2}(x, y) = \frac{x}{1 + x\,\exp(y)}
"""

from __future__ import annotations

import torch
from torch import nn

from boundlab.expr._core import Expr
from boundlab.gradlin import gradlin
from boundlab.linearop._einsum import EinsumOp
from . import ZonoBounds, _register_linearizer


def softmax2(x, y):
    return torch.nan_to_num(
        x / (1 + x * torch.exp(y)),
        nan=0.0
    )

def softmax2dx(x, y):
    return torch.nan_to_num(
        1 / (1 + x * torch.exp(y))**2,
        nan=0.0
    )

def softmax2dy(x, y):
    return -softmax2(x, y) * (1 - softmax2(x, y) / x)

def softmax2dy_inv(x, lamy, sign=1):
    # Domain for real inverse: x > 0, lamy in [-x/4, 0).
    # Near lamy -> 0-, the direct quadratic expression suffers cancellation
    # (especially for sign=1). Use asymptotic branches in that regime.
    disc = (x * (4 * lamy + x)).clamp(min=0.0)
    root = torch.sqrt(disc)

    den = 2 * lamy * x
    ratio = -((2 * lamy + x + sign * root) / den)

    near0 = lamy.abs() <= 1e-8
    if sign == 1:
        # small-root branch: exp(y) ~ -lamy / x^2
        ratio_asym = (-lamy) / (x * x)
    else:
        # large-root branch: exp(y) ~ 1 / (-lamy)
        ratio_asym = 1.0 / (-lamy)

    ratio = torch.where(near0, ratio_asym, ratio)
    finfo = torch.finfo(ratio.dtype)
    ratio = torch.nan_to_num(ratio, nan=finfo.tiny, posinf=finfo.max, neginf=finfo.tiny)
    ratio = ratio.clamp(min=finfo.tiny, max=finfo.max)
    return torch.log(ratio)


def softmax2_ub(lamx: torch.Tensor, lamy: torch.Tensor, x_ub: torch.Tensor, x_lb: torch.Tensor, y_ub: torch.Tensor, y_lb: torch.Tensor) -> torch.Tensor:
    assert torch.isfinite(lamx).all(), "softmax2_ub: lamx must be finite"
    assert torch.isfinite(lamy).all(), "softmax2_ub: lamy must be finite"

    ypos_ub = softmax2dy_inv(x_ub, lamy, sign=-1)
    ypos_lb = softmax2dy_inv(x_lb, lamy, sign=-1)
    sqrt_lamx = torch.sqrt(lamx)
    lambda0 = 1 / sqrt_lamx - 1
    # print(f"lambda0: {lambda0.item():.12g}")
    yi_ub = torch.log(lambda0 / x_ub)
    yi_lb = torch.log(lambda0 / x_lb)
    
    ypos_ub = torch.minimum(ypos_ub, yi_ub)
    ypos_ub = torch.clamp(ypos_ub, y_lb, y_ub)
    ypos_lb = torch.maximum(ypos_lb, yi_lb)
    ypos_lb = torch.clamp(ypos_lb, y_lb, y_ub)
    yi_ub = torch.clamp(yi_ub, y_lb, y_ub)
    yi_lb = torch.clamp(yi_lb, y_lb, y_ub)
 
    def f(y):
        # lambda0 / (1 + lambda0) = 1 - sqrt(lam), numerically safer near lam -> 0.
        x = lambda0 * torch.exp(-y)
        return torch.where((y <= yi_ub) & (yi_ub >= y_lb + 1e-12), softmax2(x_ub, y) - lamx * x_ub,
               torch.where((y >= yi_lb) & (yi_lb <= y_ub - 1e-12), softmax2(x_lb, y) - lamx * x_lb,
                        softmax2(x, y) - lamx * x))

    ub = torch.stack([
        f(ypos_ub) - lamy * ypos_ub,
        f(ypos_lb) - lamy * ypos_lb,
        f(yi_lb) - lamy * yi_lb,
        f(yi_ub) - lamy * yi_ub,
        f(y_lb) - lamy * y_lb,
        f(y_ub) - lamy * y_ub,
    ], dim=0).max(dim=0).values
    # print(f"ypos_ub: {ypos_ub.item():.12g}, ypos_ub value: {f(ypos_ub).item() - lamy.item() * ypos_ub.item():.12g}")
    # print(f"ypos_lb: {ypos_lb.item():.12g}, ypos_lb value: {f(ypos_lb).item() - lamy.item() * ypos_lb.item():.12g}")
    # print(f"yi_ub: {yi_ub.item():.12g}, yi_ub value: {f(yi_ub).item() - lamy.item() * yi_ub.item():.12g}")
    # print(f"yi_lb: {yi_lb.item():.12g}, yi_lb value: {f(yi_lb).item() - lamy.item() * yi_lb.item():.12g}")

    # torch.where evaluates both branches; clamp lamy for the ub2 helper branch
    # to avoid assertion failures when lamx is not in the fallback region.
    lamy_ub2 = torch.clamp(lamy, min=-1.0 + 1e-8, max=-1e-8)
    ub = torch.where(
        lamx.abs() <= 1e-8,
        torch.maximum(
            softmax2_ub2(lamy_ub2, x_ub, y_ub, y_lb) - lamx * x_ub,
            softmax2_ub2(lamy_ub2, x_lb, y_ub, y_lb) - lamx * x_lb,
        ),
        ub
    )

    assert torch.isfinite(ub).all(), "softmax2_ub: output became non-finite"
    return ub

def softmax2_ub2(lam: torch.Tensor, x: torch.Tensor, y_ub: torch.Tensor, y_lb: torch.Tensor) -> torch.Tensor:
    assert torch.isfinite(lam).all(), "softmax2_ub2: lam must be finite"
    assert ((lam <= 0.0) & (lam >= -1.0)).all(), "softmax2_ub2: lam must satisfy -1 < lam < 0"

    ypos = softmax2dy_inv(x, lam, sign=-1)
    ypos = torch.where(torch.isfinite(ypos), ypos, y_lb)
    ypos = torch.clamp(ypos, y_lb, y_ub)

    ub = torch.stack([
        softmax2(x, ypos) - lam * ypos,
        softmax2(x, y_ub) - lam * y_ub,
        softmax2(x, y_lb) - lam * y_lb,
    ]).max(dim=0).values

    assert torch.isfinite(ub).all(), "softmax2_ub2: output became non-finite"
    return ub
    
def softmax2_lb(lam: torch.Tensor, x: torch.Tensor, y_ub: torch.Tensor, y_lb: torch.Tensor) -> torch.Tensor:
    assert torch.isfinite(lam).all(), "softmax2_lb: lam must be finite"
    assert ((lam <= 0.0) & (lam >= -1.0)).all(), "softmax2_lb: lam must satisfy 0 < lam < 1"

    yneg = softmax2dy_inv(x, lam, sign=1)
    yneg = torch.where(torch.isfinite(yneg), yneg, y_ub)
    yneg = torch.clamp(yneg, y_lb, y_ub)

    lb = torch.stack([
        softmax2(x, yneg) - lam * yneg,
        softmax2(x, y_lb) - lam * y_lb,
        softmax2(x, y_ub) - lam * y_ub,
    ]).min(dim=0).values
    assert torch.isfinite(lb).all(), "softmax2_lb: output became non-finite"
    return lb


@_register_linearizer("Softmax2")
def softmax2_linearizer(
    x_ub: torch.Tensor,
    x_lb: torch.Tensor,
    y_ub: torch.Tensor,
    y_lb: torch.Tensor,
    niters = 20,
) -> ZonoBounds:
    """Gradlin-based linearizer for ``x / (1 + x * exp(y))``."""
    # Exact singleton box: return an exact affine form with zero error.
    if ((x_ub == x_lb) & (y_ub == y_lb)).all():
        x = x_ub
        y = y_ub
        lamx = softmax2dx(x, y)
        lamy = softmax2dy(x, y)
        mu = softmax2(x, y) - lamx * x - lamy * y
        beta = torch.zeros_like(mu)
        return ZonoBounds(
            bias=mu,
            error_coeffs=EinsumOp.from_hardmard(beta, len(x_ub.shape)),
            input_weights=[lamx, lamy],
        )

    x_center = (x_ub + x_lb) / 2
    y_center = (y_ub + y_lb) / 2
    lamx = nn.Parameter(
        softmax2dx(x_center, y_center),
    )
    lamy = nn.Parameter(
        softmax2dy(x_center, y_center),
    )
    gradlin_optimizer = torch.optim.LBFGS([lamx, lamy], max_iter=niters)
    
    for _ in range(niters):
        with torch.no_grad():
            lamx.nan_to_num_(nan=1e-6, posinf=1e-6, neginf=1e-6)
            lamy.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            lamx.clamp_(min=1e-12, max=1.0 - 1e-8)
            lamy.clamp_(min=-1.0 + 1e-8, max=-1e-8)
        assert torch.isfinite(lamx).all(), "softmax2_linearizer: lamx non-finite before LBFGS step"
        assert torch.isfinite(lamy).all(), "softmax2_linearizer: lamy non-finite before LBFGS step"

        def closure():
            gradlin_optimizer.zero_grad()
            with torch.no_grad():
                lamx.nan_to_num_(nan=1e-6, posinf=1e-6, neginf=1e-6)
                lamy.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                lamx.clamp_(min=1e-12, max=1.0 - 1e-8)
                lamy.clamp_(min=-1.0 + 1e-8, max=-1e-8)
            assert torch.isfinite(lamx).all(), "softmax2_linearizer: lamx non-finite in closure"
            assert torch.isfinite(lamy).all(), "softmax2_linearizer: lamy non-finite in closure"

            ub = softmax2_ub(lamx, lamy, x_ub, x_lb, y_ub, y_lb)
            lb = torch.minimum(
                softmax2_lb(lamy, x_ub, y_ub, y_lb) - lamx * x_ub,
                softmax2_lb(lamy, x_lb, y_ub, y_lb) - lamx * x_lb,
            )
            assert torch.isfinite(ub).all() and torch.isfinite(lb).all(), \
                "softmax2_linearizer: ub/lb non-finite in closure"
            bad = ub < lb
            if bad.any():
                mid = (ub + lb) / 2
                ub = torch.where(bad, mid, ub)
                lb = torch.where(bad, mid, lb)
            loss = (ub - lb).mean()
            assert torch.isfinite(loss), "softmax2_linearizer: loss non-finite"
            loss.backward()
            return loss
        gradlin_optimizer.step(closure)

    lamx = lamx.detach()
    lamy = lamy.detach()
    lamx = torch.nan_to_num(lamx, nan=1e-6, posinf=1e-6, neginf=1e-6)
    lamy = torch.nan_to_num(lamy, nan=-1e-3, posinf=-1e-8, neginf=-1.0 + 1e-8)
    lamx = torch.clamp(lamx, min=1e-12, max=1.0 - 1e-8)
    lamy = torch.clamp(lamy, min=-1.0 + 1e-8, max=-1e-8)
    ub = softmax2_ub(lamx, lamy, x_ub, x_lb, y_ub, y_lb)
    lb = torch.minimum(
        softmax2_lb(lamy, x_ub, y_ub, y_lb) - lamx * x_ub,
        softmax2_lb(lamy, x_lb, y_ub, y_lb) - lamx * x_lb,
    )
    bad = ub < lb
    if bad.any():
        mid = (ub + lb) / 2
        ub = torch.where(bad, mid, ub)
        lb = torch.where(bad, mid, lb)
    beta = (ub - lb) / 2
    mu = (ub + lb) / 2

    return ZonoBounds(
        bias=mu,
        error_coeffs=EinsumOp.from_hardmard(beta, len(x_ub.shape)),
        input_weights=[lamx, lamy],
    )


def softmax2_ibp(
    x_ub: torch.Tensor,
    x_lb: torch.Tensor,
    y_ub: torch.Tensor,
    y_lb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval bound for ``softmax2`` on a box.

    For ``x > 0``, softmax2 is monotone increasing in ``x`` and decreasing
    in ``y``:
      ub = f(x_ub, y_lb), lb = f(x_lb, y_ub).
    """
    ub = softmax2(x_ub, y_lb)
    lb = softmax2(x_lb, y_ub)
    return ub, lb


def softmax2_handler(x: Expr, y: Expr) -> Expr:
    assert x.shape == y.shape, f"softmax2 expects matching shapes, got {x.shape} vs {y.shape}"
    from . import interpret
    return interpret["Softmax2"](x, y)
