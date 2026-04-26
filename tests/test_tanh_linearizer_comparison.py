"""Comparison tests between ``tanh_linearizer_original`` and ``tanh_linearizer``.

Both functions return :class:`ZonoBounds` with

    bias=mu, error_coeffs=EinsumOp.from_hardmard(beta), input_weights=[slope]

representing the affine relaxation

    slope * x + mu - beta  <=  tanh(x)  <=  slope * x + mu + beta   for x in [lb, ub].

For each test case we:
1. Run both linearizers on the same ``(ub, lb)``.
2. Check soundness — sampled ``tanh(x)`` lies inside the affine band.
3. Compare shapes / structure of the returned ``ZonoBounds``.
4. Quantify tightness via the average band width.
"""

from __future__ import annotations

import pytest
import torch

from boundlab.zono import ZonoBounds
from boundlab.zono.tanh import tanh_linearizer, tanh_linearizer_original


def _extract(b: ZonoBounds) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (slope, mu, beta) from a :class:`ZonoBounds`."""
    assert isinstance(b, ZonoBounds)
    assert len(b.input_weights) == 1, "tanh linearizer should have a single input"
    slope = b.input_weights[0]
    mu = b.bias
    beta = b.error_coeffs.tensor
    return slope, mu, beta


def _check_sound(
    bounds: ZonoBounds, ub: torch.Tensor, lb: torch.Tensor, tol: float = 1e-5
) -> None:
    """Verify the affine band returned by ``bounds`` covers tanh on ``[lb, ub]``."""
    slope, mu, beta = _extract(bounds)
    # Sample uniformly inside [lb, ub] (broadcast across a sample axis).
    n_samples = 256
    u = torch.rand(n_samples, *ub.shape)
    x = lb.unsqueeze(0) + u * (ub - lb).unsqueeze(0)
    f = torch.tanh(x)
    upper = slope * x + mu + beta
    lower = slope * x + mu - beta
    assert (f <= upper + tol).all(), (
        f"upper bound violated by {(f - upper).max():.3e} (slope={slope}, mu={mu}, beta={beta})"
    )
    assert (f >= lower - tol).all(), (
        f"lower bound violated by {(lower - f).max():.3e} (slope={slope}, mu={mu}, beta={beta})"
    )
    # And the endpoints — the relaxation must cover them too.
    for endpoint in (lb, ub):
        f_e = torch.tanh(endpoint)
        upper_e = slope * endpoint + mu + beta
        lower_e = slope * endpoint + mu - beta
        assert (f_e <= upper_e + tol).all()
        assert (f_e >= lower_e - tol).all()


# Cases span dead/active/crossing regimes; the singleton case is treated separately
# (see ``test_singleton_*``) because the two implementations diverge there.
_CASES = [
    pytest.param(torch.tensor([1.0]), torch.tensor([-1.0]), id="symmetric_small"),
    pytest.param(torch.tensor([2.0, 3.0]), torch.tensor([-2.0, -3.0]), id="symmetric_wider"),
    pytest.param(torch.tensor([0.5]), torch.tensor([0.1]), id="positive_only"),
    pytest.param(torch.tensor([-0.1]), torch.tensor([-0.5]), id="negative_only"),
    pytest.param(torch.tensor([4.0]), torch.tensor([2.0]), id="saturated_positive"),
    pytest.param(torch.tensor([-2.0]), torch.tensor([-4.0]), id="saturated_negative"),
    pytest.param(torch.tensor([0.5]), torch.tensor([-0.5]), id="crossing"),
]


@pytest.mark.parametrize("ub, lb", _CASES)
def test_original_is_sound(ub: torch.Tensor, lb: torch.Tensor):
    bounds = tanh_linearizer_original(ub, lb)
    _check_sound(bounds, ub, lb)


@pytest.mark.parametrize("ub, lb", _CASES)
def test_new_is_sound(ub: torch.Tensor, lb: torch.Tensor):
    bounds = tanh_linearizer(ub, lb)
    _check_sound(bounds, ub, lb)


@pytest.mark.parametrize("ub, lb", _CASES)
def test_shapes_match(ub: torch.Tensor, lb: torch.Tensor):
    """Both linearizers must return same-shaped ``slope/mu/beta`` tensors."""
    s_o, m_o, b_o = _extract(tanh_linearizer_original(ub, lb))
    s_n, m_n, b_n = _extract(tanh_linearizer(ub, lb))
    assert s_o.shape == s_n.shape == ub.shape
    assert m_o.shape == m_n.shape == ub.shape
    assert b_o.shape == b_n.shape == ub.shape


# ---- singleton (ub == lb) ---------------------------------------------------
#
# Both implementations set ``mu = tanh(lb) - slope * lb`` so the affine line
# passes through ``(lb, tanh(lb))``; ``beta == 0`` since the input is exact.

_SINGLETON_LB = torch.tensor([0.3])
_SINGLETON_UB = _SINGLETON_LB.clone()


def test_singleton_new_is_sound():
    bounds = tanh_linearizer(_SINGLETON_UB, _SINGLETON_LB)
    _check_sound(bounds, _SINGLETON_UB, _SINGLETON_LB)
    _, _, beta = _extract(bounds)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-8)


def test_singleton_original_is_sound():
    bounds = tanh_linearizer_original(_SINGLETON_UB, _SINGLETON_LB)
    _check_sound(bounds, _SINGLETON_UB, _SINGLETON_LB)
    _, _, beta = _extract(bounds)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-8)


def test_singleton_implementations_match():
    """At a singleton both implementations should produce the same affine line."""
    s_o, m_o, b_o = _extract(tanh_linearizer_original(_SINGLETON_UB, _SINGLETON_LB))
    s_n, m_n, b_n = _extract(tanh_linearizer(_SINGLETON_UB, _SINGLETON_LB))
    assert torch.allclose(s_o, s_n, atol=1e-6)
    assert torch.allclose(m_o, m_n, atol=1e-6)
    assert torch.allclose(b_o, b_n, atol=1e-8)


def test_singleton_at_zero_both_match():
    """Both linearizers coincide on the ``lb == ub == 0`` singleton."""
    z = torch.zeros(3)
    s_o, m_o, b_o = _extract(tanh_linearizer_original(z, z))
    s_n, m_n, b_n = _extract(tanh_linearizer(z, z))
    assert torch.allclose(s_o, s_n, atol=1e-6)
    assert torch.allclose(m_o, m_n, atol=1e-6)
    assert torch.allclose(b_o, b_n, atol=1e-8)
    assert torch.allclose(b_o, torch.zeros_like(b_o), atol=1e-8)


@pytest.mark.parametrize("ub, lb", _CASES)
def test_band_widths_comparable(ub: torch.Tensor, lb: torch.Tensor):
    """Both relaxations should yield comparable band widths.

    The relaxation width at a point is ``2 * beta`` (the affine slope/intercept
    is the same across the band). We expect both methods to be within a small
    constant factor of each other on these well-conditioned inputs.
    """
    _, _, beta_o = _extract(tanh_linearizer_original(ub, lb))
    _, _, beta_n = _extract(tanh_linearizer(ub, lb))
    if torch.equal(ub, lb):
        assert torch.allclose(beta_o, beta_n, atol=1e-8)
        return
    # Absolute slack: the new method may be slightly looser due to the
    # softmax2 detour, but on these bounded ranges should not blow up.
    max_o = beta_o.max().item()
    max_n = beta_n.max().item()
    assert max_n <= max(10 * max_o, 1e-3), (
        f"new beta unexpectedly large: max_new={max_n:.3e}, max_orig={max_o:.3e}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_inputs_both_sound(seed: int):
    """Both linearizers stay sound across randomized intervals."""
    torch.manual_seed(seed)
    n = 16
    center = torch.randn(n)
    half = torch.rand(n) * 1.5 + 1e-3
    lb = center - half
    ub = center + half
    _check_sound(tanh_linearizer_original(ub, lb), ub, lb)
    _check_sound(tanh_linearizer(ub, lb), ub, lb)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_inputs_slope_signs(seed: int):
    """tanh has positive derivative everywhere — both slopes must be >= 0."""
    torch.manual_seed(seed)
    n = 16
    center = torch.randn(n)
    half = torch.rand(n) * 1.5 + 1e-3
    lb = center - half
    ub = center + half

    s_o, _, _ = _extract(tanh_linearizer_original(ub, lb))
    s_n, _, _ = _extract(tanh_linearizer(ub, lb))
    assert (s_o >= -1e-8).all()
    assert (s_n >= -1e-8).all()
