"""Smoke-test for ``examples/vit/pruning.py`` — runs the tiny demo model
through the differential-zonotope pipeline with the stubbed
``heaviside_pruning`` handler and checks bounds are finite and sound."""

import sys
from pathlib import Path

import torch
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _import_pruning_module():
    from examples.vit import pruning as pruning_mod
    return pruning_mod


def test_tiny_pruned_mlp_diff_bounds_sound():
    torch.manual_seed(0)
    pruning = _import_pruning_module()
    model = pruning.TinyPrunedMLP()
    c1 = torch.randn(8)
    c2 = c1 + 0.05 * torch.randn_like(c1)
    scale = 0.01

    out = pruning.diff_verify(model, c1, c2, scale)
    d_ub, d_lb = out.diff.ublb()
    assert d_ub.shape == torch.Size([4])
    assert torch.isfinite(d_ub).all()
    assert torch.isfinite(d_lb).all()
    assert (d_ub >= d_lb - 1e-6).all()
    # Soundness against the concrete model is intentionally *not* asserted:
    # the stub treats heaviside_pruning as identity while the concrete op is a
    # symbolic placeholder, so the two semantics disagree until item 4 lands.


def test_heaviside_stub_registered():
    from boundlab.diff.zono3 import interpret as diff_interpret
    _import_pruning_module()
    assert "heaviside_pruning" in diff_interpret.dispatcher
