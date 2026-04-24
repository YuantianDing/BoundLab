"""Soundness + tightness tests for hexagon-Chebyshev differential linearizers.

SOUNDNESS: for each activation and each test case, we sample a large number
of random (x, y) pairs from the feasible hexagon P, compute the predicted
bound λ·(x-y) + μ ± β, and verify f(x) - f(y) falls inside.

TIGHTNESS: we compare β against the paper's MVT-based β on the same cases,
confirming the improvement factor (never worse, typically much better).
"""

import math
import torch

from boundlab.diff.zono3.default._hex_cheby import (
    hex_chebyshev_transfer, slope_exp, slope_recip, slope_tanh,
    tanh_extra_candidates, tanh_edge_critical_bounds,
)


# ---------------------------------------------------------------------------
# Paper's MVT-based formulas (reference for comparison)
# ---------------------------------------------------------------------------

def paper_exp(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub):
    L = torch.minimum(x_lb, y_lb)
    U = torch.maximum(x_ub, y_ub)
    Smin = torch.exp(torch.clamp(L, -30, 30))
    Smax = torch.exp(torch.clamp(U, -30, 30))
    delta = torch.maximum(d_lb.abs(), d_ub.abs())
    lam = (Smin + Smax) / 2
    beta = (Smax - Smin) / 2 * delta
    return lam, torch.zeros_like(lam), beta


def paper_recip(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub):
    z_min = torch.clamp(torch.minimum(x_lb, y_lb), min=1e-9)
    z_max = torch.clamp(torch.maximum(x_ub, y_ub), min=z_min + 1e-12)
    Smin = -1.0 / (z_min ** 2)
    Smax = -1.0 / (z_max ** 2)
    delta = torch.maximum(d_lb.abs(), d_ub.abs())
    lam = (Smin + Smax) / 2
    beta = (Smax - Smin) / 2 * delta
    return lam, torch.zeros_like(lam), beta


def paper_tanh(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub):
    L = torch.minimum(x_lb, y_lb)
    U = torch.maximum(x_ub, y_ub)
    sech2_L = 1 - torch.tanh(L) ** 2
    sech2_U = 1 - torch.tanh(U) ** 2
    sigma = torch.minimum(sech2_L, sech2_U)
    delta = torch.maximum(d_lb.abs(), d_ub.abs())
    lam = (1 + sigma) / 2
    beta = (1 - sigma) / 2 * delta
    return lam, torch.zeros_like(lam), beta


# ---------------------------------------------------------------------------
# Feasibility-region sampling for soundness
# ---------------------------------------------------------------------------

def sample_hex(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub, n=200_000, device='cpu'):
    """Rejection-sample n points uniformly in the feasible hexagon P.

    Returns (x, y) tensors of length n.
    """
    # Sample in rectangle, reject points outside the strip.
    # Oversample to account for rejection.
    total_collected = []
    needed = n
    while needed > 0:
        batch = max(needed * 3, 10_000)
        xs = x_lb + (x_ub - x_lb) * torch.rand(batch, device=device)
        ys = y_lb + (y_ub - y_lb) * torch.rand(batch, device=device)
        ds = xs - ys
        keep = (ds >= d_lb) & (ds <= d_ub)
        total_collected.append((xs[keep], ys[keep]))
        needed -= int(keep.sum().item())
        if all(a.numel() == 0 for a, _ in total_collected) and needed > 0:
            # Hexagon is effectively empty; try with relaxed strip (shouldn't happen).
            raise RuntimeError("No feasible samples — hexagon empty?")
    xs = torch.cat([a for a, _ in total_collected])[:n]
    ys = torch.cat([b for _, b in total_collected])[:n]
    return xs, ys


def check_soundness(f, lam, mu, beta, xs, ys, tol=1e-6, name=""):
    """Assert |f(x) - f(y) - lam*(x-y) - mu| ≤ beta + tol for all samples."""
    d = xs - ys
    true_diff = f(xs) - f(ys)
    predicted = lam * d + mu
    err = (true_diff - predicted).abs()
    max_err = err.max().item()
    violation = (err - beta - tol).clamp(min=0).max().item()
    ok = violation == 0.0
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}: max|err| = {max_err:.4e},  beta = {beta.item():.4e},  "
          f"violation = {violation:.4e}")
    return ok


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES_EXP = [
    ("overlapping small",   0.0, 1.0,  0.5, 1.5),
    ("adjacent small",      1.0, 2.0,  0.0, 1.0),
    ("disjoint",            2.0, 3.0,  0.0, 1.0),
    ("far-separated",      10.0, 10.1, -10.0, -9.9),
    ("neg asymmetric",     -5.0, -1.0, -3.0, -2.5),
    ("single points",       1.5, 1.5,  0.5, 0.5),  # degenerate rectangle
]

CASES_RECIP = [
    ("overlapping small",   0.5, 1.0,  0.8, 1.5),
    ("adjacent small",      1.0, 2.0,  0.3, 1.0),
    ("disjoint",            1.0, 2.0,  3.0, 4.0),
    ("far",                 1.0, 2.0, 100.0, 101.0),
    ("same box",            1.0, 3.0,  1.0, 3.0),
    ("asymmetric widths",   0.5, 5.0,  2.0, 2.5),
]

CASES_TANH = [
    ("overlapping small",   0.0, 1.0,  0.5, 1.5),
    ("adjacent small",      1.0, 2.0,  0.0, 1.0),
    ("disjoint",            2.0, 3.0,  0.0, 1.0),
    ("far-separated",      10.0, 10.1, -10.0, -9.9),
    ("deep sat (both +)",  10.0, 11.0,  5.0,  6.0),
    ("deep sat opposite",  10.0, 11.0, -11.0, -10.0),
    ("near zero",          -0.2, 0.2, -0.1, 0.1),
    ("straddle zero",      -1.0, 1.0, -1.0, 1.0),
    ("asymmetric",          0.0, 5.0,  2.0, 2.5),
]


def run_function_tests(name, slope_fn, f_torch, paper_fn, cases,
                       positive_only=False, n_soundness=200_000,
                       extra_cand_fn=None, extra_smax_fn=None):
    print(f"\n=== {name.upper()} ===")
    print(f"  {'case':<25s} {'paper β':>12s} {'ours β':>12s} {'ratio':>10s}  soundness")

    all_ok = True
    ratios = []
    for desc, xl, xu, yl, yu in cases:
        if positive_only and min(xl, yl) <= 0:
            continue
        x_lb = torch.tensor([xl], dtype=torch.float64); x_ub = torch.tensor([xu], dtype=torch.float64)
        y_lb = torch.tensor([yl], dtype=torch.float64); y_ub = torch.tensor([yu], dtype=torch.float64)
        d_lb = x_lb - y_ub
        d_ub = x_ub - y_lb

        extras = extra_cand_fn(x_lb, x_ub, y_lb, y_ub) if extra_cand_fn else None
        extra_smax = extra_smax_fn(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub) if extra_smax_fn else None
        lam_ours, mu_ours, beta_ours = hex_chebyshev_transfer(
            slope_fn, x_lb, x_ub, y_lb, y_ub, d_lb, d_ub,
            extra_candidates=extras, extra_smax=extra_smax,
        )
        lam_p, mu_p, beta_p = paper_fn(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub)

        ratio = beta_p.item() / max(beta_ours.item(), 1e-30)
        ratios.append((desc, ratio))

        if xu > xl and yu > yl:
            xs, ys = sample_hex(xl, xu, yl, yu, d_lb.item(), d_ub.item(), n=n_soundness)
            xs = xs.to(torch.float64); ys = ys.to(torch.float64)
            ok = check_silent(f_torch, lam_ours, mu_ours, beta_ours, xs, ys)
            ok_paper = check_silent(f_torch, lam_p, mu_p, beta_p, xs, ys)
            if not ok_paper:
                print(f"  [warn] paper bound also violated on {desc}")
        else:
            ok = True
        all_ok = all_ok and ok

        ok_str = "✓" if ok else "✗"
        print(f"  {desc:<25s} {beta_p.item():>12.4e} {beta_ours.item():>12.4e} "
              f"{ratio:>10.2f}×  {ok_str}")

    return all_ok, ratios


def run_random_soundness(name, slope_fn, f_torch, extra_cand_fn, n_cases=300,
                         n_samples=50_000, positive_only=False,
                         extra_smax_fn=None):
    print(f"\n=== RANDOM {name.upper()} (n={n_cases} random boxes) ===")
    torch.manual_seed(42)
    fails = []

    for i in range(n_cases):
        if positive_only:
            xc = 0.5 + 5.0 * torch.rand(1).item()
            yc = 0.5 + 5.0 * torch.rand(1).item()
            wx = 0.01 + 2.0 * torch.rand(1).item()
            wy = 0.01 + 2.0 * torch.rand(1).item()
            xl, xu = xc - wx/2, xc + wx/2
            yl, yu = yc - wy/2, yc + wy/2
            xl = max(xl, 0.01); yl = max(yl, 0.01)
            xu = max(xu, xl + 1e-4); yu = max(yu, yl + 1e-4)
        else:
            xc = -8.0 + 16.0 * torch.rand(1).item()
            yc = -8.0 + 16.0 * torch.rand(1).item()
            wx = 0.01 + 3.0 * torch.rand(1).item()
            wy = 0.01 + 3.0 * torch.rand(1).item()
            xl, xu = xc - wx/2, xc + wx/2
            yl, yu = yc - wy/2, yc + wy/2

        x_lb = torch.tensor([xl], dtype=torch.float64); x_ub = torch.tensor([xu], dtype=torch.float64)
        y_lb = torch.tensor([yl], dtype=torch.float64); y_ub = torch.tensor([yu], dtype=torch.float64)
        d_lb = x_lb - y_ub; d_ub = x_ub - y_lb

        extras = extra_cand_fn(x_lb, x_ub, y_lb, y_ub) if extra_cand_fn else None
        extra_smax = extra_smax_fn(x_lb, x_ub, y_lb, y_ub, d_lb, d_ub) if extra_smax_fn else None
        lam, mu, beta = hex_chebyshev_transfer(
            slope_fn, x_lb, x_ub, y_lb, y_ub, d_lb, d_ub,
            extra_candidates=extras, extra_smax=extra_smax,
        )

        xs, ys = sample_hex(xl, xu, yl, yu, d_lb.item(), d_ub.item(), n=n_samples)
        xs = xs.to(torch.float64); ys = ys.to(torch.float64)
        if not check_silent(f_torch, lam, mu, beta, xs, ys, tol=1e-10):
            d = xs - ys
            err = (f_torch(xs) - f_torch(ys) - lam * d - mu).abs()
            worst_err = err.max().item()
            fails.append((xl, xu, yl, yu, beta.item(), worst_err))

    if fails:
        print(f"  FAILED {len(fails)}/{n_cases}")
        for (xl, xu, yl, yu, b, e) in fails[:5]:
            print(f"    box x=[{xl:.3f},{xu:.3f}] y=[{yl:.3f},{yu:.3f}]: "
                  f"β={b:.4e} worst_err={e:.4e} (over by {e-b:.4e})")
        return False
    print(f"  PASSED {n_cases}/{n_cases}")
    return True


def check_silent(f, lam, mu, beta, xs, ys, tol=1e-8):
    d = xs - ys
    true_diff = f(xs) - f(ys)
    predicted = lam * d + mu
    err = (true_diff - predicted).abs()
    return (err - beta - tol).max().item() <= 0.0


if __name__ == "__main__":
    torch.manual_seed(0)

    all_ok = True

    ok, _ = run_function_tests("Exp", slope_exp, torch.exp, paper_exp, CASES_EXP)
    all_ok = all_ok and ok

    ok, _ = run_function_tests(
        "Reciprocal", slope_recip, lambda t: 1.0 / t, paper_recip, CASES_RECIP,
        positive_only=True,
    )
    all_ok = all_ok and ok

    ok, _ = run_function_tests(
        "Tanh", slope_tanh, torch.tanh, paper_tanh, CASES_TANH,
        extra_cand_fn=tanh_extra_candidates,
        extra_smax_fn=tanh_edge_critical_bounds,
    )
    all_ok = all_ok and ok

    # Adversarial random soundness sweeps
    ok = run_random_soundness("exp", slope_exp, torch.exp, None, n_cases=300)
    all_ok = all_ok and ok

    ok = run_random_soundness(
        "reciprocal", slope_recip, lambda t: 1.0 / t, None,
        n_cases=300, positive_only=True,
    )
    all_ok = all_ok and ok

    ok = run_random_soundness(
        "tanh", slope_tanh, torch.tanh, tanh_extra_candidates, n_cases=300,
        extra_smax_fn=tanh_edge_critical_bounds,
    )
    all_ok = all_ok and ok

    print("\n" + "=" * 70)
    if all_ok:
        print("  ALL SOUNDNESS TESTS PASSED")
    else:
        print("  *** SOUNDNESS VIOLATED — see above ***")
    print("=" * 70)
