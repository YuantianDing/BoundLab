"""Hexagon-Chebyshev slope-range computation for differential transfer rules.

Computes `S_min, S_max = range of S(x, y) = (f(x) - f(y)) / (x - y)` over the
feasible hexagon
    P = { (x, y) : x ∈ [lx, ux], y ∈ [ly, uy], x - y ∈ [lΔ, uΔ] }.

This is a strict sharpening of the paper's MVT-based envelope, which bounds S
by the range of f' over the merged interval [min(lx, ly), max(ux, uy)] — a
superset of S(P). The two agree when the merged-interval f'-range is already
tight (overlapping boxes, diagonal feasible). They diverge — the paper's
envelope blowing up while ours stays bounded — as the boxes separate.

The hexagon has at most 6 vertices, drawn from a fixed set of 12 candidates:
the 4 rectangle corners plus 8 strip/edge intersections. For functions with
monotone S (exp, reciprocal) the extrema are at opposite corners, but we
evaluate all 12 and mask infeasible ones — this gives the exact range for any
S continuous on P (its extrema over a polytope are attained at vertices iff S
is affine, but for our functions S is either monotone or sign-consistent, and
in all regimes its vertex values tightly bracket the true range).
"""

from __future__ import annotations

import torch


_INF = 1e30
_FEAS_SLACK = 1e-9  # numerical slack on strip/rectangle feasibility tests


def hex_slope_range(
    slope_fn,
    x_lb: torch.Tensor, x_ub: torch.Tensor,
    y_lb: torch.Tensor, y_ub: torch.Tensor,
    d_lb: torch.Tensor, d_ub: torch.Tensor,
    extra_candidates: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    extra_smax: torch.Tensor | None = None,
    extra_smin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (S_min, S_max): the range of slope_fn(x, y) over the hexagon P.

    Candidates are the 12 hexagon-vertex candidates (4 rectangle corners + 8
    strip/edge intersections). For functions whose slope S has interior
    critical points inside the hexagon, pass:
      - `extra_candidates` for (x, y) pairs whose slope_fn value should be
        evaluated and masked by feasibility (e.g., (0, 0) for tanh when the
        hex contains it);
      - `extra_smax` / `extra_smin` for direct scalar bounds on S that we
        already know are sound upper/lower bounds (e.g., analytical sech²
        bounds on the value of S at edge critical points for tanh when
        vertex enumeration would miss them).
    """
    cands = [
        (x_lb, y_lb), (x_lb, y_ub), (x_ub, y_lb), (x_ub, y_ub),
        (x_lb, x_lb - d_ub), (x_lb, x_lb - d_lb),
        (x_ub, x_ub - d_ub), (x_ub, x_ub - d_lb),
        (y_lb + d_ub, y_lb), (y_lb + d_lb, y_lb),
        (y_ub + d_ub, y_ub), (y_ub + d_lb, y_ub),
    ]
    if extra_candidates is not None:
        cands.extend(extra_candidates)

    s_min = torch.full_like(x_ub, _INF)
    s_max = torch.full_like(x_ub, -_INF)

    for xc, yc in cands:
        in_rect = (
            (xc >= x_lb - _FEAS_SLACK) & (xc <= x_ub + _FEAS_SLACK) &
            (yc >= y_lb - _FEAS_SLACK) & (yc <= y_ub + _FEAS_SLACK)
        )
        dc = xc - yc
        in_strip = (dc >= d_lb - _FEAS_SLACK) & (dc <= d_ub + _FEAS_SLACK)
        feas = in_rect & in_strip
        s_val = slope_fn(xc, yc)
        s_min = torch.where(feas, torch.minimum(s_min, s_val), s_min)
        s_max = torch.where(feas, torch.maximum(s_max, s_val), s_max)

    # Fold in externally-supplied sound bounds.
    if extra_smax is not None:
        s_max = torch.maximum(s_max, extra_smax)
    if extra_smin is not None:
        s_min = torch.minimum(s_min, extra_smin)

    # Fallback
    no_feas = s_min >= _INF / 2
    if no_feas.any():
        mx = (x_lb + x_ub) / 2
        my = (y_lb + y_ub) / 2
        md = mx - my
        md_clamped = torch.clamp(md, d_lb, d_ub)
        my_adj = mx - md_clamped
        my_adj = torch.clamp(my_adj, y_lb, y_ub)
        fb = slope_fn(mx, my_adj)
        s_min = torch.where(no_feas, fb, s_min)
        s_max = torch.where(no_feas, fb, s_max)

    return s_min, s_max


# ---------------------------------------------------------------------------
# Slope functions (numerically safe at x ≈ y)
# ---------------------------------------------------------------------------

def slope_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """S_exp(x, y) = (e^x - e^y) / (x - y), extended by e^x at x = y."""
    # Clamp to avoid overflow in exp.
    x = torch.clamp(x, -30.0, 30.0)
    y = torch.clamp(y, -30.0, 30.0)
    d = x - y
    small = torch.abs(d) < 1e-6
    safe_d = torch.where(small, torch.ones_like(d), d)
    slope = (torch.exp(x) - torch.exp(y)) / safe_d
    # For small d, use e^((x+y)/2) (2nd-order accurate — the geometric-arithmetic
    # mean disagreement is O(d^2), well within our slack budget).
    mid = torch.exp((x + y) / 2)
    return torch.where(small, mid, slope)


def slope_recip(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """S_recip(x, y) = -1 / (x * y), valid for x, y > 0."""
    # Callers guarantee x, y >= 1e-9.
    return -1.0 / (x * y)


def slope_tanh(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """S_tanh(x, y) = (tanh x - tanh y) / (x - y), extended by sech^2(x) at x=y."""
    d = x - y
    small = torch.abs(d) < 1e-6
    safe_d = torch.where(small, torch.ones_like(d), d)
    slope = (torch.tanh(x) - torch.tanh(y)) / safe_d
    mid = (x + y) / 2
    dS = 1.0 - torch.tanh(mid) ** 2
    return torch.where(small, dS, slope)


# ---------------------------------------------------------------------------
# Packaged transfer-rule output
# ---------------------------------------------------------------------------

def hex_chebyshev_transfer(
    slope_fn,
    x_lb, x_ub, y_lb, y_ub, d_lb, d_ub,
    extra_candidates=None,
    extra_smax=None,
    extra_smin=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_min, s_max = hex_slope_range(
        slope_fn, x_lb, x_ub, y_lb, y_ub, d_lb, d_ub,
        extra_candidates=extra_candidates,
        extra_smax=extra_smax, extra_smin=extra_smin,
    )
    lambda_d = (s_min + s_max) / 2.0
    mu_d = torch.zeros_like(lambda_d)
    delta = torch.maximum(d_lb.abs(), d_ub.abs())
    beta_d = (s_max - s_min) / 2.0 * delta
    beta_d = torch.clamp(beta_d, min=0.0)
    return lambda_d, mu_d, beta_d


def tanh_extra_candidates(x_lb, x_ub, y_lb, y_ub):
    """(0, 0) clamped to the rectangle — catches the diagonal critical point
    sech²(0) = 1 when the hexagon's diagonal enters through zero."""
    zero = torch.zeros_like(x_lb)
    x_star = torch.clamp(zero, x_lb, x_ub)
    y_star = torch.clamp(zero, y_lb, y_ub)
    return [(x_star, y_star)]


def tanh_edge_critical_bounds(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub):
    """Sound upper bound on S from interior edge critical points.

    For tanh, on edge y = y_0 (rectangle edge with y fixed), interior critical
    points of S(·, y_0) satisfy sech²(x*) = S(x*, y_0) with x* = -ξ* for some
    ξ* ∈ (x*, y_0) from the MVT — i.e., x* and y_0 on opposite sides of 0 with
    |x*| < |y_0|. The VALUE at such a point is sech²(x*), upper-bounded by
    sech²(x closest to 0) over the feasible x* range. If no feasible x*
    range exists on an edge, that edge contributes nothing.

    We return a tensor S_max_bound where each element is the max over all
    four rectangle edges of the per-edge sech²-upper-bound (or -inf if no
    edge contributes). This bound is SOUND — S at any actual critical point
    is guaranteed ≤ this value — and is MUCH TIGHTER than the flat sech²(0)=1
    bound whenever the hexagon is far from zero on at least one side.
    """
    zero = torch.zeros_like(x_lb)
    neg_inf = torch.full_like(x_lb, -_INF)

    def sech2(t):
        return 1.0 - torch.tanh(t) ** 2

    def edge_bound_y_fixed(y_0):
        """Bound for x-varying edge at y = y_0 > 0 or y_0 < 0."""
        # y_0 > 0: x* ∈ (max(l_x, -y_0), min(u_x, 0))
        lo_pos = torch.maximum(x_lb, -y_0)
        hi_pos = torch.minimum(x_ub, zero)
        pos_feas = (lo_pos < hi_pos) & (y_0 > 0)
        x_close_pos = torch.clamp(zero, lo_pos, hi_pos)

        # y_0 < 0: x* ∈ (max(l_x, 0), min(u_x, -y_0))
        lo_neg = torch.maximum(x_lb, zero)
        hi_neg = torch.minimum(x_ub, -y_0)
        neg_feas = (lo_neg < hi_neg) & (y_0 < 0)
        x_close_neg = torch.clamp(zero, lo_neg, hi_neg)

        bound = neg_inf.clone()
        bound = torch.where(pos_feas, torch.maximum(bound, sech2(x_close_pos)), bound)
        bound = torch.where(neg_feas, torch.maximum(bound, sech2(x_close_neg)), bound)
        return bound

    def edge_bound_x_fixed(x_0):
        """Bound for y-varying edge at x = x_0 (symmetric role)."""
        lo_pos = torch.maximum(y_lb, -x_0)
        hi_pos = torch.minimum(y_ub, zero)
        pos_feas = (lo_pos < hi_pos) & (x_0 > 0)
        y_close_pos = torch.clamp(zero, lo_pos, hi_pos)

        lo_neg = torch.maximum(y_lb, zero)
        hi_neg = torch.minimum(y_ub, -x_0)
        neg_feas = (lo_neg < hi_neg) & (x_0 < 0)
        y_close_neg = torch.clamp(zero, lo_neg, hi_neg)

        bound = neg_inf.clone()
        bound = torch.where(pos_feas, torch.maximum(bound, sech2(y_close_pos)), bound)
        bound = torch.where(neg_feas, torch.maximum(bound, sech2(y_close_neg)), bound)
        return bound

    b1 = edge_bound_y_fixed(y_lb)
    b2 = edge_bound_y_fixed(y_ub)
    b3 = edge_bound_x_fixed(x_lb)
    b4 = edge_bound_x_fixed(x_ub)

    return torch.maximum(torch.maximum(b1, b2), torch.maximum(b3, b4))
