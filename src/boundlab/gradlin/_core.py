"""Batched trapezoid linearization for unary functions.

This module now focuses on a simpler problem:

    bound f(x) - f(y) - lam_x * x - lam_y * y

over the batched trapezoid

    lx <= x <= ux,
    ly <= y <= uy,
    ld <= x - y <= ud.

The workflow is intentionally two-stage:

1. Sample many feasible points inside each trapezoid and estimate a good
   slope pair ``(lam_x, lam_y)``. By default we use a batched Gurobi LP fit
   on those samples. Callers can opt out and fall back to a batched least
   squares fit.
2. Freeze the slopes and use Adam to search the feasible region for the
   extremal residuals, producing ``L`` and ``U`` such that

       L <= f(x) - f(y) - lam_x * x - lam_y * y <= U.

All major tensor work is batched and vectorized.
"""

from __future__ import annotations

from typing import Callable

import torch


def trapezoid_region(
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build box bounds + diff-constraint rows for the 2D trapezoid.

    Returns
    -------
    lb, ub : ``(*batch, 2)`` axis-aligned box bounds.
    A_extra, b_extra : ``(*batch, 2, 2)``, ``(*batch, 2)`` for
        ``-x + y <= -ld`` and ``x - y <= ud``.
    """
    device, dtype = lx.device, lx.dtype
    lb = torch.stack([lx, ly], dim=-1)
    ub = torch.stack([ux, uy], dim=-1)
    rows = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype)
    A_extra = rows.expand(*lx.shape, 2, 2)
    b_extra = torch.stack([-ld, ud], dim=-1)
    return lb, ub, A_extra, b_extra


def _flatten_batch(*tensors: torch.Tensor) -> tuple[list[torch.Tensor], torch.Size]:
    batch_shape = tensors[0].shape[:-1]
    return [t.reshape(-1) for t in tensors], batch_shape


def _sample_feasible_points(
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
    num_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample feasible ``(x, y)`` pairs from each batched trapezoid."""
    x_lo = torch.maximum(lx, ly + ld)
    x_hi = torch.minimum(ux, uy + ud)
    x_w = (x_hi - x_lo).clamp_min(0.0)

    # If the region is feasible, x in [x_lo, x_hi] guarantees a non-empty y-interval.
    rx = torch.rand(num_samples, *lx.shape, device=lx.device, dtype=lx.dtype, generator=generator)
    x = x_lo.unsqueeze(0) + rx * x_w.unsqueeze(0)

    y_lo = torch.maximum(ly.unsqueeze(0), x - ud.unsqueeze(0))
    y_hi = torch.minimum(uy.unsqueeze(0), x - ld.unsqueeze(0))
    ry = torch.rand(x.shape, device=lx.device, dtype=lx.dtype, generator=generator)
    y = y_lo + ry * (y_hi - y_lo).clamp_min(0.0)
    return x, y


def _sample_unconstrained_starts(
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
    num_starts: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = _sample_feasible_points(lx, ux, ly, uy, ld, ud, num_starts, generator=generator)
    return x, y


def _safe_logit01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(x) - torch.log1p(-x)


def _fit_lam_lstsq(
    fx: torch.Tensor,
    fy: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Least-squares fallback for batched slope fitting."""
    target = (fx - fy).transpose(0, 1).unsqueeze(-1)  # (*batch, S, 1)
    A = torch.stack([x.transpose(0, 1), y.transpose(0, 1)], dim=-1)  # (*batch, S, 2)
    sol = torch.linalg.lstsq(A, target).solution.squeeze(-1)
    return sol


def _fit_lam_gurobi(
    fx: torch.Tensor,
    fy: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor | None:
    """LP fit for ``lam_x, lam_y`` using Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None

    target = (fx - fy).transpose(0, 1)  # (*batch, S)
    x_t = x.transpose(0, 1)
    y_t = y.transpose(0, 1)
    batch = target.shape[0]
    out = []

    for b in range(batch):
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = 0
        lamx = model.addVar(lb=-GRB.INFINITY, name="lamx")
        lamy = model.addVar(lb=-GRB.INFINITY, name="lamy")
        t = model.addVar(lb=0.0, name="t")
        for i in range(target.shape[1]):
            ri = float(target[b, i].item())
            xi = float(x_t[b, i].item())
            yi = float(y_t[b, i].item())
            model.addConstr(ri - lamx * xi - lamy * yi <= t)
            model.addConstr(-(ri - lamx * xi - lamy * yi) <= t)
        model.setObjective(t, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            return None
        out.append((float(lamx.X), float(lamy.X)))

    return torch.tensor(out, device=fx.device, dtype=fx.dtype)


def _score_lam(
    fx: torch.Tensor,
    fy: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    residual = (fx - fy) - lam[..., 0].unsqueeze(0) * x - lam[..., 1].unsqueeze(0) * y
    return residual.amax(dim=0) - residual.amin(dim=0)


def _estimate_lam(
    f: Callable[[torch.Tensor], torch.Tensor],
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
    *,
    num_samples: int,
    lam_init: torch.Tensor | None,
    use_gurobi: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = _sample_feasible_points(lx, ux, ly, uy, ld, ud, num_samples, generator=generator)
    fx = f(x)
    fy = f(y)

    lam = _fit_lam_gurobi(fx, fy, x, y) if use_gurobi else None
    if lam is None:
        lam = _fit_lam_lstsq(fx, fy, x, y)

    if lam_init is not None:
        lam_init = lam_init.reshape_as(lam).to(device=lam.device, dtype=lam.dtype)
        lam_score = _score_lam(fx, fy, x, y, lam).reshape(-1)
        init_score = _score_lam(fx, fy, x, y, lam_init).reshape(-1)
        improved = (init_score < lam_score).unsqueeze(-1)
        lam = torch.where(improved, lam_init, lam)

    return lam, x, y, fx - fy


def _feasible_xy_from_params(
    sx: torch.Tensor,
    sy: torch.Tensor,
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_lo = torch.maximum(lx, ly + ld)
    x_hi = torch.minimum(ux, uy + ud)
    x = x_lo.unsqueeze(0) + torch.sigmoid(sx) * (x_hi - x_lo).clamp_min(1e-6).unsqueeze(0)

    y_lo = torch.maximum(ly.unsqueeze(0), x - ud.unsqueeze(0))
    y_hi = torch.minimum(uy.unsqueeze(0), x - ld.unsqueeze(0))
    y = y_lo + torch.sigmoid(sy) * (y_hi - y_lo).clamp_min(0.0)
    return x, y


def _search_extremum(
    f: Callable[[torch.Tensor], torch.Tensor],
    lam: torch.Tensor,
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
    *,
    num_starts: int,
    iters: int,
    lr: float,
    maximize: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    x0, y0 = _sample_unconstrained_starts(lx, ux, ly, uy, ld, ud, num_starts, generator=generator)
    x_lo = torch.maximum(lx, ly + ld)
    x_hi = torch.minimum(ux, uy + ud)
    sx = torch.nn.Parameter(_safe_logit01((x0 - x_lo.unsqueeze(0)) / (x_hi - x_lo).clamp_min(1e-6).unsqueeze(0)))

    y_lo0 = torch.maximum(ly.unsqueeze(0), x0 - ud.unsqueeze(0))
    y_hi0 = torch.minimum(uy.unsqueeze(0), x0 - ld.unsqueeze(0))
    sy = torch.nn.Parameter(_safe_logit01((y0 - y_lo0) / (y_hi0 - y_lo0).clamp_min(1e-6)))

    opt = torch.optim.Adam([sx, sy], lr=lr)
    best = None
    beta = 20.0
    patience = 5
    stale = 0
    tol = 1e-4

    lamx = lam[..., 0]
    lamy = lam[..., 1]

    for _ in range(iters):
        opt.zero_grad()
        x, y = _feasible_xy_from_params(sx, sy, lx, ux, ly, uy, ld, ud)
        resid = f(x) - f(y) - lamx.unsqueeze(0) * x - lamy.unsqueeze(0) * y
        if maximize:
            smooth = torch.logsumexp(beta * resid, dim=0) / beta
            loss = -smooth.mean()
        else:
            smooth = -torch.logsumexp(-beta * resid, dim=0) / beta
            loss = smooth.mean()
        loss.backward()
        opt.step()

        hard = resid.detach().amax(dim=0) if maximize else resid.detach().amin(dim=0)
        if best is None:
            best = hard
        elif maximize:
            improved = (hard > best + tol).any().item()
            best = torch.maximum(best, hard)
            stale = 0 if improved else stale + 1
        else:
            improved = (hard < best - tol).any().item()
            best = torch.minimum(best, hard)
            stale = 0 if improved else stale + 1

        if stale >= patience:
            break

    x, y = _feasible_xy_from_params(sx, sy, lx, ux, ly, uy, ld, ud)
    resid = f(x) - f(y) - lamx.unsqueeze(0) * x - lamy.unsqueeze(0) * y
    hard = resid.detach().amax(dim=0) if maximize else resid.detach().amin(dim=0)
    if best is None:
        best = hard
    elif maximize:
        best = torch.maximum(best, hard)
    else:
        best = torch.minimum(best, hard)
    return best


def gradlin(
    f: Callable[[torch.Tensor], torch.Tensor],
    lx: torch.Tensor,
    ux: torch.Tensor,
    ly: torch.Tensor,
    uy: torch.Tensor,
    ld: torch.Tensor,
    ud: torch.Tensor,
    *,
    num_samples: int = 16,
    num_starts: int = 1,
    lam_init: torch.Tensor | None = None,
    use_gurobi: bool = True,
    iters: int = 40,
    lr: float = 0.05,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate ``lam_x, lam_y, L, U`` for a batched trapezoid.

    Parameters
    ----------
    f
        Unary function accepting tensors of shape ``(*batch, S)``.
    lx, ux, ly, uy, ld, ud
        Batched trapezoid parameters with shape ``(*batch,)``.
    num_samples
        Number of sample points used to estimate the slopes.
    num_starts
        Number of Adam restarts used when searching for the extrema.
    lam_init
        Optional warm start for ``(lam_x, lam_y)`` with shape ``(*batch, 2)``.

    Returns
    -------
    lam : ``(*batch, 2)``
    L, U : ``(*batch,)``
    """
    lam, _, _, _ = _estimate_lam(
        f, lx, ux, ly, uy, ld, ud,
        num_samples=num_samples,
        lam_init=lam_init,
        use_gurobi=use_gurobi,
        generator=generator,
    )

    L = _search_extremum(
        f, lam, lx, ux, ly, uy, ld, ud,
        num_starts=num_starts, iters=iters, lr=lr, maximize=False, generator=generator,
    )
    U = _search_extremum(
        f, lam, lx, ux, ly, uy, ld, ud,
        num_starts=num_starts, iters=iters, lr=lr, maximize=True, generator=generator,
    )
    return lam, L, U
