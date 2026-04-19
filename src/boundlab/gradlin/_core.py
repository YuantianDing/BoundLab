"""Gradient-descent-based linear bound tightening.

For a smooth function ``f: R^n -> R`` and a convex polytope region
``R = { x : lb <= x <= ub, A_extra x <= b_extra }``, find ``lam in R^n``
and ``L, U in R`` such that

    lam . x + L  <=  f(x)  <=  lam . x + U    for all x in R.

The tightest constants for a fixed ``lam`` are

    U(lam) = max_{x in R} [ f(x) - lam . x ]
    L(lam) = min_{x in R} [ f(x) - lam . x ]

Candidate points for these extrema:
  * all polytope vertices (intersections of ``n`` active constraints);
  * every interior stationary point returned by ``grad_inv(lam)`` that
    lies in R (``grad_inv`` may return multiple candidates when the
    gradient equation has multiple solutions, e.g. the two branches of
    ``sech^2(y) = c`` for ``tanh``);
  * axis-face candidates: each interior candidate with one coordinate
    replaced by its box bound. For separable ``f`` these coincide with
    the true axis-face criticals; for non-separable ``f`` they are
    in-region points that don't introduce unsound tightening.

Extrema on faces that aren't axis-aligned (e.g. the ``x - y = c`` face
for bilinear ``x*y``) are **not** enumerated — pass the region as a box
(make the diff bounds loose enough to never bind) if correctness matters
for such ``f``.

The outer optimisation over ``lam`` uses Adam on the gap ``U - L``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import torch


def _enumerate_vertices(A: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate candidate vertices of the polytope ``{x : A x <= b}``.

    Takes every size-``n`` subset of the ``m`` constraint rows, solves for the
    intersection point, and checks feasibility against all ``m`` constraints.

    Parameters
    ----------
    A : ``(*batch, m, n)``
    b : ``(*batch, m)``

    Returns
    -------
    vertices : ``(*batch, V, n)`` where ``V = C(m, n)``
    feasible : ``(*batch, V)`` bool mask
    """
    m, n = A.shape[-2], A.shape[-1]
    device, dtype = A.device, A.dtype

    subsets = list(combinations(range(m), n))
    idx = torch.tensor(subsets, dtype=torch.long, device=device)  # (V, n)

    A_sub = A[..., idx, :]           # (*batch, V, n, n)
    b_sub = b[..., idx]              # (*batch, V, n)

    det = torch.linalg.det(A_sub)     # (*batch, V)
    regular = det.abs() > 1e-10

    eye = torch.eye(n, device=device, dtype=dtype).expand_as(A_sub)
    safe_A = torch.where(regular[..., None, None], A_sub, eye)
    safe_b = torch.where(regular[..., None], b_sub, torch.zeros_like(b_sub))

    vertices = torch.linalg.solve(safe_A, safe_b.unsqueeze(-1)).squeeze(-1)  # (*batch, V, n)

    Ax = torch.einsum("...mn,...vn->...vm", A, vertices)  # (*batch, V, m)
    feasible = regular & (Ax <= b.unsqueeze(-2) + 1e-6).all(dim=-1)

    return vertices, feasible


def _build_full_polytope(
    lb: torch.Tensor,
    ub: torch.Tensor,
    A_extra: torch.Tensor | None,
    b_extra: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = lb.shape[-1]
    batch_shape = lb.shape[:-1]
    device, dtype = lb.device, lb.dtype

    eye = torch.eye(n, device=device, dtype=dtype)
    box_A = torch.cat([-eye, eye], dim=0)          # (2n, n)
    box_A = box_A.expand(*batch_shape, 2 * n, n)
    box_b = torch.cat([-lb, ub], dim=-1)           # (*batch, 2n)

    if A_extra is None:
        return box_A.contiguous(), box_b.contiguous()
    return torch.cat([box_A, A_extra], dim=-2), torch.cat([box_b, b_extra], dim=-1)


def _axis_face_candidates(
    x_crit: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    """Produce ``K * 2n`` coordinate-replacement candidates.

    For each of ``K`` interior candidates, for each dimension ``i`` and
    each box bound (``lb[i]`` / ``ub[i]``), emit the candidate with the
    ``i``-th coordinate replaced. For separable ``f`` these are exact
    axis-face critical points.

    Parameters
    ----------
    x_crit : ``(*batch, K, n)``

    Returns
    -------
    candidates : ``(*batch, K * 2n, n)``
    """
    n = x_crit.shape[-1]
    x_exp = x_crit.unsqueeze(-2)                                # (*batch, K, 1, n)
    eye = torch.eye(n, dtype=torch.bool, device=x_crit.device)  # (n, n)
    lb_bc = lb.unsqueeze(-2).unsqueeze(-2)                      # (*batch, 1, 1, n)
    ub_bc = ub.unsqueeze(-2).unsqueeze(-2)
    lo = torch.where(eye, lb_bc, x_exp)                         # (*batch, K, n, n)
    hi = torch.where(eye, ub_bc, x_exp)
    stacked = torch.cat([lo, hi], dim=-2)                       # (*batch, K, 2n, n)
    return stacked.flatten(-3, -2)                              # (*batch, K*2n, n)


def _evaluate_bounds(
    f: Callable[[torch.Tensor], torch.Tensor],
    grad_inv: Callable[[torch.Tensor], torch.Tensor],
    A: torch.Tensor,
    b: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    lam: torch.Tensor,
    vertices: torch.Tensor,
    vertex_feasible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``L(lam), U(lam)`` from all candidate points."""
    n = lam.shape[-1]

    # Vertex candidates
    f_v = f(vertices)  # (*batch, V)
    h_v = f_v - (vertices * lam.unsqueeze(-2)).sum(dim=-1)

    # Interior critical(s). grad_inv may return (*batch, n) or (*batch, K, n).
    x_crit = grad_inv(lam)
    if x_crit.dim() == lam.dim():
        x_crit = x_crit.unsqueeze(-2)                                         # (*batch, 1, n)
    Ax_crit = torch.einsum("...mn,...kn->...km", A, x_crit)
    crit_feasible = (Ax_crit <= b.unsqueeze(-2) + 1e-6).all(dim=-1) & torch.isfinite(x_crit).all(dim=-1)
    h_crit = f(x_crit) - (x_crit * lam.unsqueeze(-2)).sum(dim=-1)             # (*batch, K)

    # Axis-face candidates via coordinate replacement on each interior candidate.
    x_face = _axis_face_candidates(x_crit, lb, ub)                            # (*batch, K*2n, n)
    Ax_face = torch.einsum("...mn,...kn->...km", A, x_face)
    face_feasible = (Ax_face <= b.unsqueeze(-2) + 1e-6).all(dim=-1) & torch.isfinite(x_face).all(dim=-1)
    h_face = f(x_face) - (x_face * lam.unsqueeze(-2)).sum(dim=-1)

    h_all = torch.cat([h_v, h_crit, h_face], dim=-1)
    mask_all = torch.cat([vertex_feasible, crit_feasible, face_feasible], dim=-1)

    neg_inf = torch.full_like(h_all, float("-inf"))
    pos_inf = torch.full_like(h_all, float("inf"))
    U = torch.where(mask_all, h_all, neg_inf).amax(dim=-1)
    L = torch.where(mask_all, h_all, pos_inf).amin(dim=-1)
    return L, U


def gradlin(
    f: Callable[[torch.Tensor], torch.Tensor],
    grad_inv: Callable[[torch.Tensor], torch.Tensor],
    lb: torch.Tensor,
    ub: torch.Tensor,
    A_extra: torch.Tensor | None = None,
    b_extra: torch.Tensor | None = None,
    iters: int = 200,
    lr: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tighten linear bounds on ``f`` over a box + extra-constraints region.

    Parameters
    ----------
    f : callable
        ``f(x)`` reducing the trailing ``n`` axis of ``x``. Must accept
        shapes ``(*batch, n)`` and ``(*batch, K, n)``.
    grad_inv : callable
        Inverse gradient: ``grad_inv(lam) = x`` with ``grad f(x) = lam``.
        Shape in/out: ``(*batch, n)``. Must accept ``(*batch, K, n)``.
        May return non-finite values when ``lam`` is outside the image
        of ``grad f``; those are masked.
    lb, ub : ``(*batch, n)`` — axis-aligned box bounds.
    A_extra, b_extra : optional extra constraints ``A_extra x <= b_extra``.
        Shapes ``(*batch, m, n)`` and ``(*batch, m)``.
    iters : number of Adam steps.
    lr : Adam learning rate.

    Returns
    -------
    lam : ``(*batch, n)``
    L, U : ``(*batch,)``
    """
    A, b = _build_full_polytope(lb, ub, A_extra, b_extra)
    batch_shape = lb.shape[:-1]
    n = lb.shape[-1]
    device, dtype = lb.device, lb.dtype

    vertices, vertex_feasible = _enumerate_vertices(A, b)

    lam = torch.zeros(*batch_shape, n, device=device, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([lam], lr=lr)

    for _ in range(iters):
        optimizer.zero_grad()
        L, U = _evaluate_bounds(f, grad_inv, A, b, lb, ub, lam, vertices, vertex_feasible)
        loss = (U - L).sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        lam_final = lam.detach()
        L, U = _evaluate_bounds(f, grad_inv, A, b, lb, ub, lam_final, vertices, vertex_feasible)
    return lam_final, L, U


def trapezoid_region(
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build box bounds + diff-constraint rows for the 2D trapezoid
    ``{(x, y) : lx<=x<=ux, ly<=y<=uy, ld<=x-y<=ud}``.

    All inputs are tensors of shape ``(*batch,)``.

    Returns
    -------
    lb, ub : ``(*batch, 2)`` — axis box bounds.
    A_extra, b_extra : ``(*batch, 2, 2), (*batch, 2)`` — the two diff rows.
    """
    device, dtype = lx.device, lx.dtype
    lb = torch.stack([lx, ly], dim=-1)
    ub = torch.stack([ux, uy], dim=-1)
    rows = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype)  # (2, 2)
    A_extra = rows.expand(*lx.shape, 2, 2)
    b_extra = torch.stack([-ld, ud], dim=-1)
    return lb, ub, A_extra, b_extra
