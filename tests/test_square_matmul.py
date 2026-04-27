import torch
import pytest

import boundlab.expr as expr
from boundlab.zono.bilinear import square_matmul


def _make_expr(center: torch.Tensor, half_width: torch.Tensor):
    return expr.ConstVal(center) + half_width * expr.LpEpsilon(list(center.shape))


def _sample_concrete(
    center_a: torch.Tensor,
    hw_a: torch.Tensor,
    center_b: torch.Tensor,
    hw_b: torch.Tensor,
    n_samples: int,
):
    eps_a = torch.rand((n_samples, *center_a.shape), dtype=center_a.dtype) * 2 - 1
    eps_b = torch.rand((n_samples, *center_b.shape), dtype=center_b.dtype) * 2 - 1
    a = center_a.unsqueeze(0) + hw_a.unsqueeze(0) * eps_a
    b = center_b.unsqueeze(0) + hw_b.unsqueeze(0) * eps_b
    return torch.matmul(a, b)


def _all_corner_signs(num_vars: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Build { -1, +1 }^num_vars as a dense matrix with shape (2^num_vars, num_vars).
    # Keep this for small exhaustive cases only.
    num_corners = 1 << num_vars
    ids = torch.arange(num_corners, dtype=torch.int64)
    bit_pos = torch.arange(num_vars, dtype=torch.int64)
    bits = ((ids.unsqueeze(1) >> bit_pos.unsqueeze(0)) & 1).to(dtype)
    return bits * 2.0 - 1.0


@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize("m,k,n", [(1, 2, 1), (2, 2, 2), (2, 3, 2)])
def test_square_matmul_exhaustive_corner_soundness(seed: int, m: int, k: int, n: int):
    """Compare against exact corner extrema for small independent-noise cases."""
    torch.manual_seed(seed)

    center_a = torch.randn(m, k) * 0.8
    center_b = torch.randn(k, n) * 0.8
    hw_a = torch.rand(m, k) * 0.6 + 0.05
    hw_b = torch.rand(k, n) * 0.6 + 0.05

    a_expr = _make_expr(center_a, hw_a)
    b_expr = _make_expr(center_b, hw_b)

    out_expr = square_matmul(a_expr, b_expr)
    ub, lb = out_expr.ublb()

    assert torch.isfinite(ub).all()
    assert torch.isfinite(lb).all()
    assert (ub >= lb - 1e-6).all()

    num_a = m * k
    num_b = k * n
    signs = _all_corner_signs(num_a + num_b, dtype=center_a.dtype)
    eps_a = signs[:, :num_a].reshape(-1, m, k)
    eps_b = signs[:, num_a:].reshape(-1, k, n)

    a = center_a.unsqueeze(0) + hw_a.unsqueeze(0) * eps_a
    b = center_b.unsqueeze(0) + hw_b.unsqueeze(0) * eps_b
    concrete = torch.matmul(a, b)
    true_ub = concrete.amax(dim=0)
    true_lb = concrete.amin(dim=0)

    # Soundness: abstract bounds must contain exact extrema.
    assert (ub >= true_ub - 1e-5).all(), (
        f"UB unsound. max(true_ub - ub)={(true_ub - ub).max().item():.6f}"
    )
    assert (lb <= true_lb + 1e-5).all(), (
        f"LB unsound. max(lb - true_lb)={(lb - true_lb).max().item():.6f}"
    )


@pytest.mark.parametrize(
    "seed,a_shape,b_shape,n_samples",
    [
        (0, (3, 4), (4, 5), 5000),
        (1, (2, 3, 4), (2, 4, 3), 4000),
        (2, (2, 2, 3, 4), (2, 2, 4, 2), 2500),
        (3, (1, 2, 5, 3), (1, 2, 3, 4), 2500),
    ],
)
def test_square_matmul_monte_carlo_stress(
    seed: int, a_shape: tuple[int, ...], b_shape: tuple[int, ...], n_samples: int
):
    """Stress test soundness on larger (including batched) shapes."""
    torch.manual_seed(seed)

    center_a = torch.randn(*a_shape) * 0.7
    center_b = torch.randn(*b_shape) * 0.7
    hw_a = torch.rand(*a_shape) * 0.5 + 0.05
    hw_b = torch.rand(*b_shape) * 0.5 + 0.05

    a_expr = _make_expr(center_a, hw_a)
    b_expr = _make_expr(center_b, hw_b)
    out_expr = square_matmul(a_expr, b_expr)
    ub, lb = out_expr.ublb()

    assert torch.isfinite(ub).all()
    assert torch.isfinite(lb).all()
    assert (ub >= lb - 1e-6).all()

    concrete = _sample_concrete(center_a, hw_a, center_b, hw_b, n_samples=n_samples)
    max_ub_violation = (concrete - ub.unsqueeze(0)).max().item()
    max_lb_violation = (lb.unsqueeze(0) - concrete).max().item()

    assert max_ub_violation <= 1e-4, f"UB unsound by {max_ub_violation:.6f}"
    assert max_lb_violation <= 1e-4, f"LB unsound by {max_lb_violation:.6f}"
