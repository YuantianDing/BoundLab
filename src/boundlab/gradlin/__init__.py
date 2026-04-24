"""Gradient-descent-based linear bound tightening.

For a smooth ``f: R^n -> R`` and a polytope region ``R = {x : A x <= b}``,
:func:`gradlin` finds ``lam in R^n`` and scalars ``L, U`` such that

    lam . x + L  <=  f(x)  <=  lam . x + U   for all x in R.

Example
-------
>>> import torch
>>> from boundlab.gradlin import gradlin, trapezoid_region
>>> lx = torch.tensor([-1.0]); ux = torch.tensor([1.0])
>>> ly = torch.tensor([-1.0]); uy = torch.tensor([1.0])
>>> ld = torch.tensor([-2.0]); ud = torch.tensor([2.0])
>>> lb, ub, A, b = trapezoid_region(lx, ux, ly, uy, ld, ud)
>>> f = lambda xy: xy[..., 0] * xy[..., 1]
>>> grad_inv = lambda lam: torch.stack([lam[..., 1], lam[..., 0]], dim=-1)
>>> lam, L, U = gradlin(f, grad_inv, lb, ub, A, b, iters=50)
"""

from ._core import gradlin, trapezoid_region

__all__ = ["gradlin", "trapezoid_region"]
