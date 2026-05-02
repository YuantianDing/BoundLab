"""Batched trapezoid linearization for unary functions.

:func:`gradlin` estimates ``(lam_x, lam_y)`` and bounds ``L, U`` for

    f(x) - f(y) - lam_x * x - lam_y * y

over the batched trapezoid

    lx <= x <= ux,
    ly <= y <= uy,
    ld <= x - y <= ud.

The solver first samples points to fit the slopes and then uses Adam to
search for extremal residuals.

Example
-------
>>> import torch
>>> from boundlab.gradlin import gradlin, trapezoid_region
>>> lx = torch.tensor([-1.0]); ux = torch.tensor([1.0])
>>> ly = torch.tensor([-1.0]); uy = torch.tensor([1.0])
>>> ld = torch.tensor([-2.0]); ud = torch.tensor([2.0])
>>> f = lambda x: torch.exp(x)
>>> lam, L, U = gradlin(f, lx, ux, ly, uy, ld, ud, num_samples=128, iters=10)
"""

from ._core import gradlin, trapezoid_region

__all__ = ["gradlin", "trapezoid_region"]
