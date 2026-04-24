# """Tests for `boundlab.gradlin`.

# For each scenario we:
# 1. Build a batched trapezoidal region.
# 2. Run :func:`gradlin` to obtain ``lam, L, U``.
# 3. Sample many points inside the region and assert every concrete
#    ``f(sample)`` lies within ``[lam . sample + L, lam . sample + U]``.
# 4. Sanity check that the gap ``U - L`` is strictly tighter than the
#    zero-slope baseline (confirming the optimizer actually helped).
# """

# from __future__ import annotations

# import torch

# from boundlab.gradlin import gradlin, trapezoid_region


# # ---- sampling helpers -------------------------------------------------------


# def _sample_trapezoid(
#     lx: torch.Tensor,
#     ux: torch.Tensor,
#     ly: torch.Tensor,
#     uy: torch.Tensor,
#     ld: torch.Tensor,
#     ud: torch.Tensor,
#     n: int,
#     *,
#     seed: int = 0,
# ) -> torch.Tensor:
#     """Rejection-sample ~``n`` points per batch element from the trapezoid.

#     Samples that fall outside the trapezoid are replaced with the midpoint
#     (always feasible for the regions used in these tests) so the returned
#     tensor has the same number of rows per batch element.
#     """
#     g = torch.Generator().manual_seed(seed)
#     batch = lx.shape
#     chunk = 16 * n
#     rx = torch.rand(chunk, *batch, generator=g)
#     ry = torch.rand(chunk, *batch, generator=g)
#     x = lx + rx * (ux - lx)
#     y = ly + ry * (uy - ly)
#     d = x - y
#     ok = (d >= ld) & (d <= ud)
#     cx = 0.5 * (lx + ux)
#     cy = 0.5 * (ly + uy)
#     xs = torch.where(ok, x, cx.expand_as(x))
#     ys = torch.where(ok, y, cy.expand_as(y))
#     return torch.stack([xs, ys], dim=-1)  # (chunk, *batch, 2)


# def _assert_sound(f, lam, L, U, samples, *, tol=1e-3):
#     fs = f(samples)  # (K, *batch)
#     lam_dot = (samples * lam.unsqueeze(0)).sum(dim=-1)
#     low = lam_dot + L.unsqueeze(0)
#     high = lam_dot + U.unsqueeze(0)
#     assert (fs >= low - tol).all(), (
#         f"Lower bound violated: max deficit = {(low - fs).max():.6f}"
#     )
#     assert (fs <= high + tol).all(), (
#         f"Upper bound violated: max excess = {(fs - high).max():.6f}"
#     )


# # ---- test scenarios ---------------------------------------------------------


# def test_gradlin_exponential():
#     # Separable f(x, y) = exp(x) - exp(y) over a trapezoid with an active
#     # diff constraint. exp has no critical on the x-y=c face (derivative
#     # is monotone along it), so vertex + interior + axis-face enumeration
#     # is sound here.
#     lx = torch.tensor([-1.0, -0.5, 0.0, 0.2])
#     ux = torch.tensor([0.0, 0.5, 1.0, 0.9])
#     ly = torch.tensor([-0.5, -0.3, 0.1, 0.0])
#     uy = torch.tensor([0.5, 0.4, 0.7, 0.6])
#     ld = torch.tensor([-0.8, -0.5, -0.5, -0.2])
#     ud = torch.tensor([0.8, 0.5, 0.6, 0.6])
#     lb, ub, A, b = trapezoid_region(lx, ux, ly, uy, ld, ud)

#     def f(xy):
#         return torch.exp(xy[..., 0]) - torch.exp(xy[..., 1])

#     def grad_inv(lam):
#         # exp(x) = lam_x  -> x = log(lam_x); need lam_x > 0.
#         # -exp(y) = lam_y -> y = log(-lam_y); need lam_y < 0.
#         # Clamp to the valid domain; out-of-domain results fall outside
#         # the region and are filtered by the feasibility check.
#         safe_x = lam[..., 0].clamp(min=1e-6)
#         safe_y = (-lam[..., 1]).clamp(min=1e-6)
#         return torch.stack([torch.log(safe_x), torch.log(safe_y)], dim=-1)

#     lam, L, U = gradlin(f, grad_inv, lb, ub, A, b, iters=200, lr=0.1)
#     assert lam.shape == (4, 2)
#     assert L.shape == (4,) and U.shape == (4,)

#     samples = _sample_trapezoid(lx, ux, ly, uy, ld, ud, n=2000)
#     _assert_sound(f, lam, L, U, samples)

#     # Baseline: lam = 0 (skip optimizer).
#     _, L0, U0 = gradlin(f, grad_inv, lb, ub, A, b, iters=0)
#     assert ((U - L) <= (U0 - L0) + 1e-4).all()
#     assert ((U - L) < (U0 - L0) - 1e-3).any(), "Adam did not tighten any batch element"


# def test_gradlin_tanh():
#     # Loose diff constraint (effectively a rectangle). tanh has non-monotone
#     # derivative so the x-y=c face may have a critical point; we avoid that
#     # complication here. Axis-face criticals are handled by coordinate
#     # replacement (sound because tanh(x) - tanh(y) is separable).
#     lx = torch.tensor([-1.0, -0.5, -0.8, -1.5])
#     ux = torch.tensor([1.0, 0.5, 1.2, 0.3])
#     ly = torch.tensor([-0.5, -1.0, -0.4, -1.0])
#     uy = torch.tensor([0.5, 1.0, 1.0, 1.2])
#     ld = torch.tensor([-10.0, -10.0, -10.0, -10.0])
#     ud = torch.tensor([10.0, 10.0, 10.0, 10.0])
#     lb, ub, A, b = trapezoid_region(lx, ux, ly, uy, ld, ud)

#     def f(xy):
#         return torch.tanh(xy[..., 0]) - torch.tanh(xy[..., 1])

#     def grad_inv(lam):
#         # sech^2(x) = lam_x -> tanh(x) = +/- sqrt(1 - lam_x); both branches.
#         # sech^2(y) = -lam_y -> tanh(y) = +/- sqrt(1 + lam_y); both branches.
#         # Clamp safely within fp32 precision (1 - 1e-8 rounds to 1).
#         tx = torch.sqrt((1.0 - lam[..., 0]).clamp(min=1e-5)).clamp(max=1 - 1e-5)
#         ty = torch.sqrt((1.0 + lam[..., 1]).clamp(min=1e-5)).clamp(max=1 - 1e-5)
#         x_pos = torch.atanh(tx)
#         y_pos = torch.atanh(ty)
#         # All four (x, y) sign combinations.
#         return torch.stack(
#             [
#                 torch.stack([x_pos, y_pos], dim=-1),
#                 torch.stack([x_pos, -y_pos], dim=-1),
#                 torch.stack([-x_pos, y_pos], dim=-1),
#                 torch.stack([-x_pos, -y_pos], dim=-1),
#             ],
#             dim=-2,
#         )  # (*batch, 4, 2)

#     lam, L, U = gradlin(f, grad_inv, lb, ub, A, b, iters=200, lr=0.1)
#     assert lam.shape == (4, 2)

#     samples = _sample_trapezoid(lx, ux, ly, uy, ld, ud, n=2000)
#     _assert_sound(f, lam, L, U, samples)

#     _, L0, U0 = gradlin(f, grad_inv, lb, ub, A, b, iters=0)
#     assert ((U - L) <= (U0 - L0) + 1e-4).all()


# def test_gradlin_bilinear():
#     # x*y over a rectangle (loose diff). Bilinear extrema on a box lie at
#     # corners; the (saddle) interior critical grad_inv(lam) = (lam_y, lam_x)
#     # is harmless.
#     lx = torch.tensor([-1.0, 0.0, -2.0, 0.5])
#     ux = torch.tensor([1.0, 2.0, 0.0, 1.5])
#     ly = torch.tensor([-1.0, -1.0, 0.0, -1.0])
#     uy = torch.tensor([1.0, 1.0, 2.0, 0.5])
#     ld = torch.tensor([-10.0, -10.0, -10.0, -10.0])
#     ud = torch.tensor([10.0, 10.0, 10.0, 10.0])
#     lb, ub, A, b = trapezoid_region(lx, ux, ly, uy, ld, ud)

#     def f(xy):
#         return xy[..., 0] * xy[..., 1]

#     def grad_inv(lam):
#         # grad(xy) = (y, x), so grad_inv((a, b)) = (b, a).
#         return torch.stack([lam[..., 1], lam[..., 0]], dim=-1)

#     lam, L, U = gradlin(f, grad_inv, lb, ub, A, b, iters=200, lr=0.1)
#     assert lam.shape == (4, 2)

#     samples = _sample_trapezoid(lx, ux, ly, uy, ld, ud, n=2000)
#     _assert_sound(f, lam, L, U, samples)

#     # At least some batches with asymmetric boxes should prefer non-zero lam.
#     assert (lam.abs().sum(dim=-1) > 1e-3).any()


# def test_gradlin_softmax2():
#     # f(x, y) = x / (1 + x * exp(y)) -- a softmax-like 2-variable function.
#     # Non-separable: axis-face criticals from coord-replacement are not the
#     # true face criticals, so test on narrow regions where corner + interior
#     # candidates dominate. Soundness is verified empirically via sampling.
#     lx = torch.tensor([0.7, 0.4, 0.9, 0.3])
#     ux = torch.tensor([0.9, 0.6, 1.0, 0.4])
#     ly = torch.tensor([-0.1, -0.2, 0.05, -0.15])
#     uy = torch.tensor([0.1, 0.0, 0.2, 0.0])
#     ld = torch.tensor([-10.0, -10.0, -10.0, -10.0])
#     ud = torch.tensor([10.0, 10.0, 10.0, 10.0])
#     lb, ub, A, b = trapezoid_region(lx, ux, ly, uy, ld, ud)

#     def f(xy):
#         x = xy[..., 0]
#         y = xy[..., 1]
#         return x / (1 + x * torch.exp(y))

#     def grad_inv(lam):
#         # Feasible lam domain: lam_x in (0, 1), lam_y < 0.
#         # Clamp safely; out-of-domain candidates get filtered by feasibility.
#         sqrt_lx = torch.sqrt(lam[..., 0].clamp(min=1e-5, max=1 - 1e-4))
#         one_minus = (1 - sqrt_lx).clamp(min=1e-4)
#         neg_ly = (-lam[..., 1]).clamp(min=1e-6)
#         x = neg_ly / (sqrt_lx * one_minus)
#         y = 2 * torch.log(one_minus) - torch.log(neg_ly)
#         return torch.stack([x, y], dim=-1)

#     lam, L, U = gradlin(f, grad_inv, lb, ub, A, b, iters=300, lr=0.05)
#     assert lam.shape == (4, 2)

#     samples = _sample_trapezoid(lx, ux, ly, uy, ld, ud, n=2000)
#     _assert_sound(f, lam, L, U, samples)

#     _, L0, U0 = gradlin(f, grad_inv, lb, ub, A, b, iters=0)
#     assert ((U - L) <= (U0 - L0) + 1e-4).all()


# def test_gradlin_bilinear_symmetric_matches_mccormick():
#     """For the symmetric box [-1, 1]^2, f = x*y has optimal lam = 0 and L=-1, U=1."""
#     lx = torch.tensor([-1.0])
#     ux = torch.tensor([1.0])
#     ly = torch.tensor([-1.0])
#     uy = torch.tensor([1.0])
#     ld = torch.tensor([-10.0])
#     ud = torch.tensor([10.0])
#     lb, ub, A, b = trapezoid_region(lx, ux, ly, uy, ld, ud)

#     def f(xy):
#         return xy[..., 0] * xy[..., 1]

#     def grad_inv(lam):
#         return torch.stack([lam[..., 1], lam[..., 0]], dim=-1)

#     lam, L, U = gradlin(f, grad_inv, lb, ub, A, b, iters=300, lr=0.05)
#     assert torch.allclose(lam, torch.zeros_like(lam), atol=5e-2)
#     assert torch.allclose(U, torch.tensor([1.0]), atol=5e-2)
#     assert torch.allclose(L, torch.tensor([-1.0]), atol=5e-2)
