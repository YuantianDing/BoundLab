"""Transformer zonotope soundness tests.

Tests zonotope abstract interpretation on transformer-like architectures:
- Linear attention (no softmax) — tests bilinear matmul handler
- Softmax attention — tests exp, reciprocal linearizers and softmax composition
- Feed-forward blocks with various activations

For each model, we:
1. Build the zonotope expression via zono.interpret.
2. Compute upper and lower bounds via bound propagation.
3. Sample many concrete inputs from the L∞ perturbation ball.
4. Assert that every concrete output lies within the computed bounds.
"""

import math

import torch
import pytest
from torch import nn
import cProfile
import pstats
from io import StringIO

import boundlab.expr as expr
from boundlab.expr._core import Expr
import boundlab.prop as prop
from boundlab.sparse.coo import MultiCOOTensorSum
import boundlab.utils as utils
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export


def _export(model: nn.Module, in_shape: list[int]):
    """Export *model* to ONNX IR."""
    return onnx_export(model, (in_shape,))


# ---- helpers ----------------------------------------------------------------

def _make_input(center_val: torch.Tensor, scale: float = 1.0):
    """Create a zonotope expression: center + scale * eps."""
    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon(list(center_val.shape))
    if scale == 1.0:
        return center + eps
    return center + eps * scale


def test_bilinear_matmul_accepts_asymmetric_residuals():
    torch.manual_seed(0)
    a_center = torch.tensor([[-0.2, 0.1, 0.4], [0.3, -0.1, 0.2]])
    b_center = torch.randn(3, 2)
    a_in = _make_input(a_center, scale=0.35)
    b_expr = _make_input(b_center, scale=0.1)
    a_expr = zono.interpret["relu"](a_in)

    out = zono.bilinear_matmul(a_expr, b_expr)
    ub, lb = out.ublb()

    noise_a = (torch.rand(500, *a_center.shape) * 2 - 1) * 0.35
    noise_b = (torch.rand(500, *b_center.shape) * 2 - 1) * 0.1
    concrete = torch.relu(a_center + noise_a) @ (b_center + noise_b)
    assert (concrete <= ub.unsqueeze(0) + 1e-5).all()
    assert (concrete >= lb.unsqueeze(0) - 1e-5).all()


def _sample_inputs(center: torch.Tensor, scale: float, n: int = 2000) -> torch.Tensor:
    """Uniform samples from the L∞ ball of radius `scale` around `center`."""
    noise = torch.rand(n, *center.shape) * 2 - 1  # ∈ (-1, 1)
    return center.unsqueeze(0) + scale * noise


def _check_bounds(
    outputs: torch.Tensor, ub: torch.Tensor, lb: torch.Tensor, tol: float = 1e-4
):
    assert (ub >= lb - tol).all(), (
        f"Upper bound < lower bound: max violation = {(lb - ub).max():.6f}"
    )
    assert (outputs <= ub.unsqueeze(0) + tol).all(), (
        f"Upper bound violated: max excess = {(outputs - ub.unsqueeze(0)).max():.6f}"
    )
    assert (outputs >= lb.unsqueeze(0) - tol).all(), (
        f"Lower bound violated: max deficit = {(lb.unsqueeze(0) - outputs).max():.6f}"
    )


# ---- Linear Attention (no softmax) ------------------------------------------

class LinearAttention(nn.Module):
    """Single-head linear attention (no softmax) + FFN block."""

    def __init__(self, d_model, d_k, d_ff):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        self.proj = nn.Linear(d_k, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        attn = torch.matmul(Q, K.transpose(0, 1))
        context = torch.matmul(attn, V)
        out = self.proj(context)
        x = x + out
        h = self.relu(self.ff1(x))
        x = x + self.ff2(h)
        return x


def test_linear_attention_sound():
    """Zonotope bounds for linear attention must contain all sampled outputs."""
    torch.manual_seed(100)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    seq_len, d_model, d_k, d_ff = 5, 4, 4, 8
    model = LinearAttention(d_model, d_k, d_ff)
    model.eval()

    center_val = torch.randn(seq_len, d_model) * 0.5
    scale = 0.1  # small perturbation for tighter bounds

    pr = cProfile.Profile()
    pr.enable()
    op = zono.interpret(_export(model, [seq_len, d_model]))

    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(f"\n{'='*70}")
    print(f"Profile: {__file__}")
    print(f"{'='*70}\n")
    print(s.getvalue())

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb)


# ---- Bilinear matmul only (Q @ K^T) ----------------------------------------

def test_bilinear_matmul_sound():
    """Bilinear matmul (A @ B where both are Expr) produces sound bounds."""
    torch.manual_seed(101)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    m, k, n = 3, 4, 2
    center_A = torch.randn(m, k)
    center_B = torch.randn(k, n)
    scale = 0.1

    A_expr = _make_input(center_A, scale=scale)
    B_expr = _make_input(center_B, scale=scale)
    C_expr = zono.bilinear_matmul(A_expr, B_expr)

    ub, lb = C_expr.ublb()

    # Monte Carlo verification
    n_samples = 3000
    for _ in range(n_samples):
        noise_A = (torch.rand(m, k) * 2 - 1) * scale
        noise_B = (torch.rand(k, n) * 2 - 1) * scale
        A_concrete = center_A + noise_A
        B_concrete = center_B + noise_B
        C_concrete = A_concrete @ B_concrete
        assert (C_concrete <= ub + 1e-4).all(), \
            f"UB violated: {(C_concrete - ub).max():.6f}"
        assert (C_concrete >= lb - 1e-4).all(), \
            f"LB violated: {(lb - C_concrete).max():.6f}"


# ---- Exp linearizer ---------------------------------------------------------

def test_exp_linearizer_sound():
    """Exp linearizer produces sound bounds."""
    torch.manual_seed(102)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    center_val = torch.randn(5) * 0.5
    scale = 0.3
    x_expr = _make_input(center_val, scale=scale)

    exp_handler = zono.interpret["exp"]
    y_expr = exp_handler(x_expr)
    ub, lb = y_expr.ublb()

    samples = _sample_inputs(center_val, scale, n=3000)
    outputs = torch.exp(samples)
    _check_bounds(outputs, ub, lb)


# ---- Reciprocal linearizer --------------------------------------------------

def test_reciprocal_linearizer_sound():
    """Reciprocal linearizer produces sound bounds for positive inputs."""
    torch.manual_seed(103)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    # Ensure strictly positive inputs
    center_val = torch.rand(5) + 1.0  # center in [1, 2]
    scale = 0.2  # perturbation won't make it negative
    x_expr = _make_input(center_val, scale=scale)

    recip_handler = zono.interpret["reciprocal"]
    y_expr = recip_handler(x_expr)
    ub, lb = y_expr.ublb()

    samples = _sample_inputs(center_val, scale, n=3000)
    outputs = 1.0 / samples
    _check_bounds(outputs, ub, lb)


# ---- Tanh linearizer --------------------------------------------------------

def test_tanh_linearizer_sound():
    """Tanh linearizer produces sound bounds."""
    torch.manual_seed(104)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    center_val = torch.randn(5) * 0.8
    scale = 0.5
    x_expr = _make_input(center_val, scale=scale)

    tanh_handler = zono.interpret["tanh"]
    y_expr = tanh_handler(x_expr)
    ub, lb = y_expr.ublb()

    samples = _sample_inputs(center_val, scale, n=3000)
    outputs = torch.tanh(samples)
    _check_bounds(outputs, ub, lb)

def test_pairwise_diff0():
    """pairwise_diff should produce x[..., i] - x[..., j]."""
    diff = utils.pairwise_diff0(expr.LpEpsilon((3,4,5)), dim=-1)
    jac1 = diff.ops[0].jacobian()

    diff = utils.pairwise_diff(expr.LpEpsilon((3,4,5)), dim=-1)
    jac2 = diff.ops[0].jacobian()

    assert torch.allclose(jac1, jac2)

# ---- Softmax handler ---------------------------------------------------------

def test_pairwise_diff_matches_expected_orientation():
    """pairwise_diff should produce x[..., i] - x[..., j]."""
    center_val = torch.tensor([[1.0, 2.0, 4.0], [-1.0, 0.5, 3.0]])
    diff = utils.pairwise_diff(expr.ConstVal(center_val), dim=-1)
    ub, lb = diff.ublb()

    expected = center_val.unsqueeze(-1) - center_val.unsqueeze(-2)
    assert torch.allclose(ub, expected)
    assert torch.allclose(lb, expected)


def test_pairwise_diff_preserves_offdiagonal_uncertainty():
    """Off-diagonal differences of uncertain inputs should stay uncertain."""
    center_val = torch.zeros(3)
    scale = 0.1
    x = _make_input(center_val, scale=scale)
    N = x.shape[-1]
    l = x.unsqueeze(-2).expand_on(-2, N)
    r = x.unsqueeze(-1).expand_on(-1, N)
    op = (l - r).ops[0]
    print(op.tensor.apply_multiplicative(lambda t: t.abs()).to_dense().tensor)
    print(op.abs().tensor.to_dense().tensor)

    ub, lb = (l - r).ublb()

    width = ub - lb
    print(width)
    offdiag = ~torch.eye(center_val.numel(), dtype=torch.bool)
    assert torch.allclose(width.diagonal(), torch.zeros(center_val.numel()))
    assert (width[offdiag] >= 4 * scale - 1e-6).all()



def test_pairwise_diff_matches_expected_orientation():
    """pairwise_diff should produce x[..., i] - x[..., j]."""
    center_val = torch.tensor([[1.0, 2.0, 4.0], [-1.0, 0.5, 3.0]])
    diff = utils.pairwise_diff(expr.ConstVal(center_val), dim=-1)
    ub, lb = diff.ublb()

    expected = center_val.unsqueeze(-1) - center_val.unsqueeze(-2)
    assert torch.allclose(ub, expected)
    assert torch.allclose(lb, expected)

def test_softmax_bounds_are_not_point_for_uncertain_input():
    """Softmax bounds should not collapse to the center value for eps inputs."""
    torch.manual_seed(105)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    center_val = torch.randn(3, 4) * 0.3
    y_expr = zono.softmax_handler(_make_input(center_val, scale=0.1), dim=-1)
    ub, lb = y_expr.ublb()

    assert ((ub - lb) > 0).any()


def test_softmax_large_pairwise_gaps_remain_finite_and_probabilistic():
    """Large logit gaps should not create NaNs or invalid probability bounds."""
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    center_val = torch.tensor([[31.0, 30.0, 0.0]])
    y_expr = zono.softmax_handler(_make_input(center_val, scale=0.1), dim=-1)
    ub, lb = y_expr.ublb()

    assert torch.isfinite(ub).all()
    assert torch.isfinite(lb).all()
    assert (ub >= lb - 1e-6).all()
    assert (lb >= -1e-6).all()
    assert (ub <= 1 + 1e-6).all()


def test_softmax_sound():
    """Softmax handler produces sound bounds."""
    torch.manual_seed(105)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    seq_len, d = 3, 4
    center_val = torch.randn(seq_len, d) * 0.3
    scale = 0.1  # small perturbation for softmax

    x_expr = _make_input(center_val, scale=scale)
    y_expr = zono.softmax_handler(x_expr, dim=-1)
    ub, lb = y_expr.ublb()

    samples = _sample_inputs(center_val, scale, n=3000)
    outputs = torch.stack([torch.softmax(s, dim=-1) for s in samples])
    _check_bounds(outputs, ub, lb, tol=1e-3)


# ---- Softmax Attention -------------------------------------------------------

class SoftmaxAttention(nn.Module):
    """Single-head attention with softmax + FFN block."""

    def __init__(self, d_model, d_k, d_ff):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        self.proj = nn.Linear(d_k, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_ff, d_model)
        self.scale = math.sqrt(d_k)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        out = self.proj(context)
        x = x + out
        h = self.relu(self.ff1(x))
        x = x + self.ff2(h)
        return x


def test_softmax_attention_sound():
    """Zonotope bounds for softmax attention must contain all sampled outputs."""
    torch.manual_seed(106)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    seq_len, d_model, d_k, d_ff = 2, 3, 2, 4
    model = SoftmaxAttention(d_model, d_k, d_ff)
    model.eval()

    center_val = torch.randn(seq_len, d_model) * 0.3
    scale = 0.05  # very small perturbation for softmax stability

    op = zono.interpret(_export(model, [seq_len, d_model]))
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    samples = _sample_inputs(center_val, scale, n=3000)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb, tol=1e-3)


class TinyThreeLayerTransformer(nn.Module):
    """Tiny stacked transformer used to exercise multi-layer interpretation."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            LinearAttention(d_model=2, d_k=2, d_ff=3) for _ in range(3)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_tiny_three_layer_transformer_sound():
    """A tiny 3-layer transformer should produce sound zonotope bounds."""
    torch.manual_seed(108)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    model = TinyThreeLayerTransformer().eval()
    center_val = torch.randn(2, 2) * 0.1
    scale = 0.01

    op = zono.interpret(_export(model, [2, 2]))
    x_expr = _make_input(center_val, scale=scale)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    samples = _sample_inputs(center_val, scale, n=512)
    with torch.no_grad():
        outputs = torch.stack([model(s) for s in samples])
    _check_bounds(outputs, ub, lb, tol=1e-3)


# ---- FFN with Tanh -----------------------------------------------------------

class TanhFFN(nn.Module):
    """FFN block using tanh activation."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def test_interpreter_tanh_ffn_sound():
    """zono.interpret on Tanh FFN produces sound bounds."""
    torch.manual_seed(107)
    prop._UB_CACHE.clear()
    prop._LB_CACHE.clear()

    model = TanhFFN(4, 6, 3)
    model.eval()
    center_val = torch.randn(4)

    op = zono.interpret(_export(model, [4]))
    x_expr = _make_input(center_val)
    y_expr = op(x_expr)
    ub, lb = y_expr.ublb()

    samples = torch.rand(2000, 4) * 2 - 1 + center_val
    with torch.no_grad():
        outputs = model(samples)
    _check_bounds(outputs, ub, lb)
