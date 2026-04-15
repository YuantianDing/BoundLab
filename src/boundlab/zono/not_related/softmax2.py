


import torch

from boundlab.zono import _register_linearizer
from boundlab.zono import ZonoBounds
from boundlab.linearop import ScalarOp
from boundlab.linearop._einsum import EinsumOp


def _softmax2_points_y(x: torch.Tensor, lam: torch.Tensor, ub: torch.Tensor, lb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower and upper bound for softmax2(x, y) - lam * y = x / (1 + x exp(y)) - lam * y.

    d/dy softmax2(x, y) = -softmax2(x, y) * (1 - softmax2(x, y) / x)
                        = -softmax2(x, y) + softmax2(x, y)^2 / x
                        = 1/x (softmax2(x, y) - x)^2 - x
                        = x (1/(1+exp(y)) - 1)^2 - x
    y = ln(1/ (1 _+ sqrt(lam / x + 1)) - 1)
    """
    ypos = torch.log(1 / (1 + torch.sqrt(lam / x + 1)) - 1)
    yneg = torch.log(1 / (1 - torch.sqrt(lam / x + 1)) - 1)

    ypos2 = torch.clamp(ypos, lb, ub)
    yneg2 = torch.clamp(yneg, lb, ub)

    def f(x, y, lam):
        return x / (1 + x * torch.exp(y)) - lam * y

    lower_point = torch.where(
        f(x, yneg2, lam) < f(x, ub, lam),
        yneg2, ub
    )
    upper_point = torch.where(
        f(x, ypos2, lam) >= f(x, lb, lam),
        ypos2, lb
    )

    return upper_point, lower_point

def _softmax2_bounds(
        lam_x: torch.Tensor, lam_y: torch.Tensor,
        ux: torch.Tensor, lx: torch.Tensor,
        uy: torch.Tensor, ly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower and upper bound for softmax2(x, y) - lam_x * x - lam_y * y = x / (1 + x exp(y)) - lam_x * x - lam_y * y.
    """

    yuu, yul = _softmax2_points_y(ux, lam_y, uy, ly)
    ylu, yll = _softmax2_points_y(lx, lam_y, uy, ly)

    def f(x, y, lam_x, lam_y):
        return x / (1 + x * torch.exp(y)) - lam_x * x - lam_y * y
    
    lam0 = 1 / torch.sqrt(lam_x) - 1
    assert torch.isfinite(lam0).all() and (lam0 > 0).all(), f"Non-finite or non-positive lam0: {lam0}"
    xuy = lam0 / torch.exp(uy)
    xly = lam0 / torch.exp(ly)
    upper_bound = torch.stack([
        f(ux, yuu, lam_x, lam_y),
        f(lx, ylu, lam_x, lam_y),
        f(xuy, uy, lam_x, lam_y),
        f(xly, ly, lam_x, lam_y),
    ]).max(dim=0).values
    lower_bound = torch.minimum(
        f(ux, yul, lam_x, lam_y),
        f(lx, yll, lam_x, lam_y),
    )
    return upper_bound, lower_bound

@_register_linearizer("Softmax2")
def softmax2_linearizer(ux: torch.Tensor, lx: torch.Tensor, uy: torch.Tensor, ly: torch.Tensor) -> ZonoBounds:
    """Linearization of softmax2(x, y) = x / (1 + x exp(y)).

    Returns (lam_x, lam_y) such that softmax2(x, y) >= lam_x * x + lam_y * y for all x in [lx, ux], y in [ly, uy].

    Examples
    --------
    >>> import torch
    >>> from boundlab.zono.softmax2 import softmax2_linearizer
    >>> ux = torch.tensor([2.0])
    >>> lx = torch.tensor([1.0])
    >>> uy = torch.tensor([0.5])
    >>> ly = torch.tensor([-0.5])
    >>> lam_x, lam_y = softmax2_linearizer(ux, lx, uy, ly)
    >>> lam_x.shape
    torch.Size([1])
    >>> lam_y.shape
    torch.Size([1])
    """
    def f(x, y):
        return x / (1.0 + x * torch.exp(y))
    def fx(x, y):
        return 1.0 / (1.0 + x * torch.exp(y)) ** 2
    def fy(x, y):
        expy = torch.exp(y)
        return -x * expy / (1.0 + x * expy) / (1.0 + x * expy)
    assert (ux > 0).all() and (lx > 0).all()

    assert torch.isfinite(ux * torch.exp(uy) / (1.0 + ux * torch.exp(uy))).all(), f"Non-finite fy(ux, uy): {fy(ux, uy)}"
    assert torch.isfinite(fy(ux, uy)).all(), f"Non-finite fy(ux, uy): {fy(ux, uy)}"
    lam_x = (fx(ux, uy) + fx(ux, ly)) / 2.0
    lam_y = (fy(ux, uy) + fy(lx, uy)) / 2.0
    assert torch.isfinite(lam_x).all()
    assert torch.isfinite(lam_y).all()

    ub, lb = _softmax2_bounds(lam_x, lam_y, ux, lx, uy, ly)
    bias = (ub + lb) / 2
    error_bound = (ub - lb) / 2
    assert torch.isfinite(bias).all() and torch.isfinite(error_bound).all(), f"Non-finite bias or error_bound: {bias}, {error_bound}"

    if error_bound.dim() == 0:
        err_op = ScalarOp(error_bound.item(), torch.Size([]))
    else:
        err_op = EinsumOp.from_hardmard(error_bound, error_bound.dim())

    return ZonoBounds(
        bias=bias,
        error_coeffs=err_op,
        input_weights=[lam_x, lam_y],
    )


# Compatibility: softmax.py expects a handler symbol; reuse the linearizer.
softmax2_handler = softmax2_linearizer
    



