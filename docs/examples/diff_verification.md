# Example: Differential Verification with `diff.zono3`

This example shows how to use `boundlab.diff.zono3` to compute tight bounds
on `f₁(x) − f₂(x)` — the output *difference* between two networks — and
compares those bounds against the naïve approach of bounding each network
independently.

## Setup

```python
import torch
from torch import nn

import boundlab.expr as expr
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.op import DiffLinear
from boundlab.diff.zono3 import interpret as diff_interpret

torch.manual_seed(0)
```

---

## Part 1 — Two Networks, Same Input Distribution

The most common use case: `f₁` and `f₂` have identical architecture but
different weights, and we want to certify how far their outputs can diverge
over an L∞ ball around a nominal input.

```python
# Shared architecture, different first-layer weights.
fc1 = nn.Linear(4, 8)
fc2 = nn.Linear(4, 8)

model = nn.Sequential(DiffLinear(fc1, fc2), nn.ReLU(), nn.Linear(8, 3))
exported = torch.export.export(model, (torch.randn(4),))
op = diff_interpret(exported)

# Nominal input; both networks see the same perturbation ball.
c = torch.randn(4)
x = expr.ConstVal(c) + 0.1 * expr.LpEpsilon([4])

out = op(x)

# After ReLU the output is promoted to DiffExpr3.
assert isinstance(out, DiffExpr3)

d_ub, d_lb = out.diff.ublb()
print("Max diff UB:", d_ub.max().item())
print("Min diff LB:", d_lb.min().item())
```

### Soundness check

```python
s = c + 0.1 * (torch.rand(2000, 4) * 2 - 1)
with torch.no_grad():
    diffs = torch.relu(fc1(s)) @ model[2].weight.T + model[2].bias \
          - torch.relu(fc2(s)) @ model[2].weight.T - model[2].bias
    diffs = nn.Sequential(nn.ReLU(), model[2])(fc1(s)) \
          - nn.Sequential(nn.ReLU(), model[2])(fc2(s))

assert (diffs <= d_ub.unsqueeze(0) + 1e-5).all()
assert (diffs >= d_lb.unsqueeze(0) - 1e-5).all()
print("Soundness: OK")
```

---

## Part 2 — Explicit `DiffExpr3` Input

When the two networks are not connected through `DiffLinear`, build a
`DiffExpr3` directly and pass it to `diff_interpret`:

```python
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
exported = torch.export.export(model, (torch.randn(4),))
op = diff_interpret(exported)

# Two distinct perturbation balls (independent epsilon symbols).
c1 = torch.randn(4)
c2 = torch.randn(4)
x = expr.ConstVal(c1) + 0.1 * expr.LpEpsilon([4])
y = expr.ConstVal(c2) + 0.1 * expr.LpEpsilon([4])
d = x - y

out = op(DiffExpr3(x, y, d))
d_ub, d_lb = out.diff.ublb()
print("Output diff width:", (d_ub - d_lb).max().item())
```

---

## Part 3 — Tighter Bounds via Shared Noise

When `x` and `y` share the **same epsilon symbol**, the differential
interpreter exploits the correlation and gives strictly tighter bounds than
running two independent zonotope passes:

```python
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
exported = torch.export.export(model, (torch.randn(4),))

c1 = torch.randn(4)
c2 = c1 + 0.15 * torch.randn(4)

# Shared epsilon — same noise object for both branches.
shared_eps = expr.LpEpsilon([4])
x = expr.ConstVal(c1) + shared_eps
y = expr.ConstVal(c2) + shared_eps

# Differential bounds (exploits shared noise).
op_diff = diff_interpret(exported)
out = op_diff(DiffExpr3(x, y, x - y))
diff_width = out.diff.bound_width()

# Naïve bounds (independent epsilon symbols, no correlation).
op_std = zono.interpret(exported)
y1 = op_std(expr.ConstVal(c1) + expr.LpEpsilon([4]))
y2 = op_std(expr.ConstVal(c2) + expr.LpEpsilon([4]))
naive_width = (y1.ub() - y2.lb()) - (y1.lb() - y2.ub())

print("Differential width:", diff_width.tolist())
print("Naïve width:       ", naive_width.tolist())
assert (diff_width < naive_width).all()
print("Tighter: OK")
```

The improvement is most pronounced when `c1 ≈ c2` (networks see nearly the
same input) and the model contains many ReLU neurons in the crossing
regime, where the joint sign-pattern analysis of the differential linearizer
pays off most.

---

## Part 4 — Regime Breakdown for a Single ReLU Neuron

This snippet illustrates the four deterministic regimes of the differential
ReLU handler:

```python
from boundlab.diff.expr import DiffExpr2, DiffExpr3
import boundlab.zono as zono

relu_handler = diff_interpret.dispatcher["relu"]

def _pt(center, half):
    """Point zonotope: 1-d, center ± half."""
    e = expr.LpEpsilon([1])
    return expr.ConstVal(torch.tensor([center])) \
         + torch.tensor([half]) * e

# dead / dead  →  diff = 0 exactly
x, y = _pt(-2.0, 0.5), _pt(-3.0, 0.5)
out = relu_handler(DiffExpr3(x, y, x - y))
print("dead/dead  diff ub:", out.diff.ub())   # tensor([0.])

# active / active  →  diff passes through unchanged
x, y = _pt(2.0, 0.5), _pt(1.0, 0.5)
out = relu_handler(DiffExpr3(x, y, x - y))
print("act/act    diff ub:", out.diff.ub())   # matches x - y exactly

# active / dead  →  diff bounded by [lb_x, ub_x]
x, y = _pt(2.0, 0.5), _pt(-2.0, 0.5)
out = relu_handler(DiffExpr3(x, y, x - y))
print("act/dead   diff ub:", out.diff.ub())   # tensor([2.5])

# dead / active  →  diff bounded by [-ub_y, -lb_y]
x, y = _pt(-2.0, 0.5), _pt(2.0, 0.5)
out = relu_handler(DiffExpr3(x, y, x - y))
print("dead/act   diff ub:", out.diff.ub())   # tensor([-1.5])
```

---

## Part 5 — Deep Network Differential Soundness

```python
torch.manual_seed(1)
fc1 = nn.Linear(6, 10)
fc2 = nn.Linear(6, 10)

model = nn.Sequential(
    DiffLinear(fc1, fc2),
    nn.ReLU(),
    nn.Linear(10, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
)
exported = torch.export.export(model, (torch.randn(6),))
op = diff_interpret(exported)

c = torch.randn(6)
x = expr.ConstVal(c) + 0.3 * expr.LpEpsilon([6])
out = op(x)
d_ub, d_lb = out.diff.ublb()

s = c + 0.3 * (torch.rand(3000, 6) * 2 - 1)
with torch.no_grad():
    fc_shared = model[2:]
    h1 = torch.relu(fc1(s))
    h2 = torch.relu(fc2(s))
    diffs = fc_shared(h1) - fc_shared(h2)

assert (diffs <= d_ub.unsqueeze(0) + 1e-4).all()
assert (diffs >= d_lb.unsqueeze(0) - 1e-4).all()
print("Deep network soundness: OK")
```

---

## What This Demonstrates

- Building `DiffExpr3` inputs with independent or shared noise symbols.
- Using `DiffLinear` to introduce weight splitting at the first layer.
- Running `diff.zono3.interpret` and extracting bounds from `out.diff`.
- The tightness advantage of differential tracking over naïve subtraction.
- Regime-level behaviour of the differential ReLU linearizer.
- End-to-end soundness verification via Monte Carlo sampling.
