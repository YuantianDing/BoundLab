"""End-to-end zonotope soundness tests.

For each network, we:
1. Build the zonotope expression manually using expression algebra.
2. Compute upper and lower bounds via bound propagation.
3. Sample many concrete inputs from the L∞ perturbation ball.
4. Assert that every concrete output lies within the computed bounds.
"""

import torch
import pytest
from torch import nn
import random

import boundlab.expr as expr
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export

# ---- helpers ----------------------------------------------------------------

def _export(model: nn.Module, in_shape: list[int]):
    """Export *model* to ONNX IR."""
    return onnx_export(model, (in_shape,))


_relu_handler = zono.interpret["relu"]


def _sample_inputs(center: torch.Tensor, n: int = 2000) -> torch.Tensor:
    """Uniform samples from the L∞ ball of radius 1 around `center`."""
    eps = torch.rand(n, *center.shape) * 2 - 1  # ∈ (-1, 1)^n
    return center.unsqueeze(0) + eps


def _check_bounds(
    outputs: torch.Tensor, ub: torch.Tensor, lb: torch.Tensor, tol: float = 1e-5
):
    assert (outputs <= ub.unsqueeze(0) + tol).all(), (
        f"Upper bound violated: max excess = {(outputs - ub.unsqueeze(0)).max():.6f}"
    )
    assert (outputs >= lb.unsqueeze(0) - tol).all(), (
        f"Lower bound violated: max deficit = {(lb.unsqueeze(0) - outputs).max():.6f}"
    )


# ---- single linear layer (bounds must be exact) ----------------------------

def test_linear_bounds_exact():
    """For a purely linear map the zonotope bounds are exact."""
    torch.manual_seed(0)
    W = torch.randn(3, 4)
    b = torch.randn(3)
    center_val = torch.randn(4)

    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon([4])
    x = expr.Add(center, eps)

    # y = W @ x + b
    y = W @ x + expr.ConstVal(b)
    ub, lb = y.ublb()

    samples = _sample_inputs(center_val)
    outputs = samples @ W.T + b
    _check_bounds(outputs, ub, lb)

    # For a linear map the bounds are exact: ub = W@center + b + |W|@1
    ub_exact = W @ center_val + b + W.abs() @ torch.ones(4)
    lb_exact = W @ center_val + b - W.abs() @ torch.ones(4)
    assert torch.allclose(ub, ub_exact, atol=1e-5), "Linear upper bound should be exact"
    assert torch.allclose(lb, lb_exact, atol=1e-5), "Linear lower bound should be exact"


# ---- linear + relu (soundness) ----------------------------------------------

def test_linear_relu_bounds_sound():
    """Zonotope bounds for Linear+ReLU must contain all sampled outputs."""
    torch.manual_seed(1)
    W = torch.randn(5, 4)
    b = torch.randn(5)
    center_val = torch.randn(4)

    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon([4])
    x = expr.Add(center, eps)

    y_linear = W @ x + expr.ConstVal(b)
    y = _relu_handler(y_linear)
    ub, lb = y.ublb()

    samples = _sample_inputs(center_val)
    outputs = torch.relu(samples @ W.T + b)
    _check_bounds(outputs, ub, lb)


# ---- all-dead neurons -------------------------------------------------------

def test_relu_all_dead():
    """When all neurons are dead (ub <= 0), bounds should be [0, 0]."""
    torch.manual_seed(2)
    # bias pushes everything to [-3, -1]
    W = torch.eye(3)
    b = torch.full((3,), -2.0)
    center_val = torch.zeros(3)

    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon([3])
    x = expr.Add(center, eps)

    y_linear = W @ x + expr.ConstVal(b)
    y = _relu_handler(y_linear)

    assert torch.allclose(y.ub(), torch.zeros(3), atol=1e-6)
    assert torch.allclose(y.lb(), torch.zeros(3), atol=1e-6)


# ---- all-active neurons -----------------------------------------------------

def test_relu_all_active():
    """When all neurons are active (lb >= 0), relu is identity — bounds exact."""
    torch.manual_seed(3)
    W = torch.eye(3)
    b = torch.full((3,), 2.0)
    center_val = torch.zeros(3)

    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon([3])
    x = expr.Add(center, eps)

    y_linear = W @ x + expr.ConstVal(b)
    y = _relu_handler(y_linear)

    assert torch.allclose(y.ub(), torch.full((3,), 3.0), atol=1e-5)
    assert torch.allclose(y.lb(), torch.full((3,), 1.0), atol=1e-5)


# ---- two-layer network (linear → relu → linear → relu) ---------------------

def test_two_layer_bounds_sound():
    """Zonotope bounds for a 2-layer MLP must contain all sampled outputs."""
    torch.manual_seed(4)
    W1 = torch.randn(6, 4)
    b1 = torch.randn(6)
    W2 = torch.randn(3, 6)
    b2 = torch.randn(3)
    center_val = torch.randn(4)

    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon([4])
    x = expr.Add(center, eps)

    # Layer 1
    h1 = _relu_handler(W1 @ x + expr.ConstVal(b1))
    # Layer 2
    h2 = _relu_handler(W2 @ h1 + expr.ConstVal(b2))

    ub, lb = h2.ublb()

    samples = _sample_inputs(center_val)
    hidden = torch.relu(samples @ W1.T + b1)
    outputs = torch.relu(hidden @ W2.T + b2)
    _check_bounds(outputs, ub, lb)


# ---- shared input (structural sharing) -------------------------------------

def test_structural_sharing_bounds_sound():
    """Two branches sharing the same input eps — shared dependency tracked."""
    torch.manual_seed(5)
    W1 = torch.randn(3, 2)
    W2 = torch.randn(3, 2)
    center_val = torch.randn(2)

    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon([2])
    x = expr.Add(center, eps)

    # Two linear branches share x
    branch1 = W1 @ x
    branch2 = W2 @ x
    y = branch1 + branch2  # equivalent to (W1 + W2) @ x
    ub, lb = y.ublb()

    samples = _sample_inputs(center_val)
    outputs = samples @ (W1 + W2).T
    _check_bounds(outputs, ub, lb)


# ---- zono.interpret (operator API) -----------------------------------------

def _make_input(center_val: torch.Tensor):
    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon(list(center_val.shape))
    return expr.Add(center, eps)


def test_interpreter_single_linear():
    """zono.interpret on a single nn.Linear produces exact bounds."""
    torch.manual_seed(10)
    model = nn.Linear(4, 3)
    center_val = torch.randn(4)

    op = zono.interpret(_export(model, [4]))
    y = op(_make_input(center_val))
    ub, lb = y.ublb()

    samples = _sample_inputs(center_val)
    with torch.no_grad():
        outputs = model(samples)
    _check_bounds(outputs, ub, lb)

    # Linear map: bounds are exact
    W = model.weight.detach()
    b = model.bias.detach()
    ub_exact = W @ center_val + b + W.abs() @ torch.ones(4)
    lb_exact = W @ center_val + b - W.abs() @ torch.ones(4)
    assert torch.allclose(ub, ub_exact, atol=1e-4), "Linear upper bound should be exact"
    assert torch.allclose(lb, lb_exact, atol=1e-4), "Linear lower bound should be exact"


def test_interpreter_linear_relu_sound():
    """zono.interpret on Linear+ReLU produces sound bounds."""
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU())
    center_val = torch.randn(4)

    op = zono.interpret(_export(model, [4]))
    y = op(_make_input(center_val))
    ub, lb = y.ublb()

    samples = _sample_inputs(center_val)
    with torch.no_grad():
        outputs = model(samples)
    _check_bounds(outputs, ub, lb, tol=1.5e-1)


def test_interpreter_two_layer_sound():
    """zono.interpret on a 2-layer MLP produces sound bounds."""
    torch.manual_seed(12)
    model = nn.Sequential(
        nn.Linear(4, 6), nn.ReLU(),
        nn.Linear(6, 3), nn.ReLU(),
    )
    center_val = torch.randn(4)

    op = zono.interpret(_export(model, [4]))
    y = op(_make_input(center_val))
    ub, lb = y.ublb()

    samples = _sample_inputs(center_val)
    with torch.no_grad():
        outputs = model(samples)
    _check_bounds(outputs, ub, lb)


def test_shared_dependency_exact_cancellation():
    """Expressions reusing the same input should cancel exactly when subtracted."""
    torch.manual_seed(13)
    center_val = torch.randn(7)
    x = _make_input(center_val)
    W = torch.randn(4, 7)
    y = (W @ x) + ((-W) @ x)
    ub, lb = y.ublb()

    zeros = torch.zeros(4)
    assert torch.allclose(ub, zeros, atol=1e-6)
    assert torch.allclose(lb, zeros, atol=1e-6)


def test_relu_mixed_regimes_piecewise_bounds():
    """ReLU should match expected bounds for dead/active/crossing neurons."""
    torch.manual_seed(14)
    center_val = torch.tensor([0.0, 2.0, -2.0])
    scale = torch.tensor([1.0, 0.5, 0.5])

    center = expr.ConstVal(center_val)
    eps = expr.LpEpsilon([3])
    x = center + scale * eps

    y = _relu_handler(x)
    ub, lb = y.ublb()

    samples = _sample_inputs(center_val, n=3000)
    outputs = torch.relu(center_val + samples.sub(center_val) * scale)
    _check_bounds(outputs, ub, lb)

    ub_exact = torch.tensor([1.0, 2.5, 0.0])
    # Crossing neuron uses triangle relaxation, so lb can be negative.
    lb_exact = torch.tensor([-0.5, 1.5, 0.0])
    assert torch.allclose(ub, ub_exact, atol=1e-5)
    assert torch.allclose(lb, lb_exact, atol=1e-5)


def test_softmax2_linearizer_sound():
    """softmax2 linearizer must enclose all sampled concrete values."""
    torch.manual_seed(15)
    from boundlab.zono.softmax2 import softmax2_linearizer

    # 1-D box domain (avoid scalar tensors).
    lx = torch.tensor([0.5])
    ux = torch.tensor([3.0])
    ly = torch.tensor([-1.0])
    uy = torch.tensor([1.0])

    zono_bounds = softmax2_linearizer(ux, lx, uy, ly)
    lam_x, lam_y = zono_bounds.input_weights
    bias = zono_bounds.bias
    ec = zono_bounds.error_coeffs
    err = torch.as_tensor(ec.scalar if hasattr(ec, "scalar") else ec.tensor, device=bias.device).reshape(bias.shape)

    def concrete_softmax2(x, y):
        return x / (1 + x * torch.exp(y))

    # Sample concrete points in the box and check they fall inside bounds.
    n = 4000
    x_samples = lx + (ux - lx) * torch.rand(n, *lx.shape)
    y_samples = ly + (uy - ly) * torch.rand(n, *ly.shape)
    vals = concrete_softmax2(x_samples, y_samples)

    centers = lam_x * x_samples + lam_y * y_samples + bias
    ub_pred = centers + err
    lb_pred = centers - err

    assert torch.all(vals <= ub_pred + 1e-6)
    assert torch.all(vals >= lb_pred - 1e-6)


def test_softmax2_linearizer_vector_boxes():
    """Vector-domain softmax2 bounds remain sound across random boxes."""
    torch.manual_seed(123)
    from boundlab.zono.softmax2 import softmax2_linearizer

    def run_once():
        # Random positive ranges for x and signed ranges for y.
        lx = torch.rand(1000) / 2
        ux = (torch.rand(1000) + 1) / 2
        ly = -1.5 + torch.rand(1000) * 0.5
        uy = ly + torch.rand(1000) * 2.0

        zb = softmax2_linearizer(ux, lx, uy, ly)
        lam_x, lam_y = zb.input_weights
        bias = zb.bias
        ec = zb.error_coeffs
        err = torch.as_tensor(ec.scalar if hasattr(ec, "scalar") else ec.tensor, device=bias.device).reshape(bias.shape)

        def concrete_softmax2(x, y):
            return x / (1 + x * torch.exp(y))

        n = 5000
        x_samples = lx + (ux - lx) * torch.rand(n, *lx.shape)
        y_samples = ly + (uy - ly) * torch.rand(n, *ly.shape)
        vals = concrete_softmax2(x_samples, y_samples)

        centers = lam_x * x_samples + lam_y * y_samples + bias
        ub_pred = centers + err
        lb_pred = centers - err

        tol = 5e-2
        assert torch.all(vals >= lb_pred - tol)
        groups = torch.nonzero(vals > ub_pred + tol)
        if len(groups) > 0:
            print(f"Found {len(groups)} violations:")
            indices = [random.randint(0, len(groups)) for _ in range(min(5, len(groups)))]
            for i in range(min(5, len(groups))):
                id0, id1 = groups[indices[i]][0], groups[indices[i]][1]
                print(f"  i={indices[i]} x={x_samples[id0, id1].item():.3f}, y={y_samples[id0, id1].item():.3f}, lx={lx[id1].item():.3f}, ux={ux[id1].item():.3f}, ly={ly[id1].item():.3f}, uy={uy[id1].item():.3f}, lam_x={lam_x[id1].item():.3f}, lam_y={lam_y[id1].item():.3f}, bias={bias[id1].item():.3f}, err={err[id1].item():.3f}, val={vals[id0, id1].item():.3f}, ub_pred={ub_pred[id0, id1].item():.3f}")
        assert torch.all(vals <= ub_pred + tol)

    for _ in range(3):
        run_once()


def test_softmax2_points_y_hits_extrema():
    """_softmax2_points_y should pick near-extreme points within bounds."""
    torch.manual_seed(321)
    from boundlab.zono.softmax2 import _softmax2_points_y

    def f(x, y, lam):
        return x / (1 + x * torch.exp(y)) - lam * y

    x = torch.rand(1000)
    lam = torch.rand(1000)
    lb = torch.rand(1000) - 2.0
    ub = torch.rand(1000) - 1.0

    upper, lower = _softmax2_points_y(x, lam, ub, lb)

    # Candidates should stay within provided bounds.
    assert torch.all(upper <= ub + 1e-8) and torch.all(upper >= lb - 1e-8)
    assert torch.all(lower <= ub + 1e-8) and torch.all(lower >= lb - 1e-8)

    # Verify they achieve near-max/min of f over a fine grid per element.
    for i in range(len(x)):
        grid = torch.linspace(lb[i], ub[i], 400)
        vals = f(x[i], grid, lam[i])
        max_val = vals.max()
        min_val = vals.min()
        assert torch.isclose(f(x[i], upper[i], lam[i]), max_val, atol=1e-3)
        assert torch.isclose(f(x[i], lower[i], lam[i]), min_val, atol=1e-3)


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_interpreter_deeper_network_sound(seed: int):
    """Stress-check soundness on deeper ReLU networks over multiple seeds."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(5, 9), nn.ReLU(),
        nn.Linear(9, 7), nn.ReLU(),
        nn.Linear(7, 6), nn.ReLU(),
        nn.Linear(6, 4), nn.ReLU(),
    )
    center_val = torch.randn(5)

    op = zono.interpret(_export(model, [5]))
    y = op(_make_input(center_val))
    ub, lb = y.ublb()

    samples = _sample_inputs(center_val, n=2500)
    with torch.no_grad():
        outputs = model(samples)
    _check_bounds(outputs, ub, lb, tol=2e-5)
