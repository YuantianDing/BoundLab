# Differential Verification with `diff.zono3`

`boundlab.diff.zono3` computes over-approximations of the *difference*
`f₁(x) − f₂(x)` between two structurally similar networks.
Tracking the difference directly — rather than bounding each network
independently and subtracting — exploits shared input noise and produces
provably tighter certificates.

## Why Differential Verification?

Consider two networks that differ only in a single weight matrix.
Standard verification bounds each output independently:

```{math}
[\mathrm{lb}_1, \mathrm{ub}_1] \quad \text{and} \quad [\mathrm{lb}_2, \mathrm{ub}_2]
```

so the naive diff bound is `[lb₁ - ub₂, ub₁ - lb₂]`, which is as wide
as the *sum* of both intervals — even when the networks are nearly identical.

Differential verification instead tracks the triple `(x, y, d)` where `d`
over-approximates `f₁(x) − f₂(x)` *jointly*.  For affine layers the bias
cancels exactly; for nonlinear layers a specialised relaxation (VeryDiff,
Teuber et al. 2024) exploits the sign structure of both branches to
tighten the error approximation.

## Core Abstractions

### `DiffExpr2`

A pair `(x, y)` tracking two network branches simultaneously.
All linear operators (`+`, `-`, `*`, `@`, shape ops, indexing) apply
element-wise to both components.

```python
from boundlab.diff.expr import DiffExpr2
import boundlab.expr as expr
import torch

x = expr.ConstVal(torch.zeros(4)) + expr.LpEpsilon([4])
y = expr.ConstVal(torch.ones(4)) + expr.LpEpsilon([4])
pair = DiffExpr2(x, y)

# All ops apply to both components:
shifted = pair + torch.tensor([1.0, 0.0, 0.0, 0.0])
scaled  = pair * 2.0
sub     = pair[1:3]          # DiffExpr2 with shape [2]
```

`DiffExpr2` is produced automatically by `diff_pair_handler` when the
interpreter encounters a `diff_pair` node.  It is promoted to `DiffExpr3`
by the first nonlinear handler (e.g., `relu`).

### `DiffExpr3`

A triple `(x, y, diff)` where `diff` over-approximates `f₁(x) − f₂(x)`.
Affine operations preserve the invariant:

- **Shared additive constant** cancels in `diff` automatically.
- **Shared weight matrix `W`** applied to both branches:
  `diff_out = W @ diff_in` (no bias contribution).
- **Different weight matrices `W₁`, `W₂`** use the bilinear identity:
  `diff_out = W₁ @ diff + (W₁ - W₂) @ y`.

```python
from boundlab.diff.expr import DiffExpr3
import boundlab.expr as expr
import torch

c1 = torch.randn(4)
c2 = torch.randn(4)
x = expr.ConstVal(c1) + expr.LpEpsilon([4])
y = expr.ConstVal(c2) + expr.LpEpsilon([4])
d = x - y   # exact diff expression

triple = DiffExpr3(x, y, d)

# Arithmetic on all three components:
neg = -triple            # negates x, y, and diff
add = triple + triple    # diffs add
```

Integer indexing (`triple[0]`, `triple[1]`, `triple[2]`) returns the
underlying component expressions (`x`, `y`, `diff`) rather than tensor
elements.  Slice indexing applies element-wise to all three:

```python
sub = triple[2:5]    # DiffExpr3 with shape [3], all components sliced
```

### `diff_pair` and `DiffLinear`

`diff_pair(x, y)` is a registered `torch.library` custom operator that
marks two tensors as a *differentially-paired* input.  At the concrete
tensor level it returns `x` unchanged (a no-op); when an exported graph is
run through `diff.zono3.interpret` the node is lifted into a `DiffExpr2`.

```python
import torch
from boundlab.diff.op import diff_pair

# Works in eager mode (returns x):
out = diff_pair(torch.zeros(4), torch.ones(4))

# Captured verbatim by torch.export:
import torch.export
class Model(torch.nn.Module):
    def forward(self, x, y):
        return diff_pair(x, y)

exported = torch.export.export(Model(), (torch.zeros(4), torch.zeros(4)))
any("diff_pair" in n.target.__name__
    for n in exported.graph.nodes if n.op == "call_function")  # True
```

`DiffLinear` wraps two `nn.Linear` layers and pairs their outputs with
`diff_pair`.  It is the recommended way to introduce differential tracking
at the first layer of a two-network comparison:

```python
from torch import nn
from boundlab.diff.op import DiffLinear

fc1 = nn.Linear(4, 8)
fc2 = nn.Linear(4, 8)  # different weights

model = nn.Sequential(DiffLinear(fc1, fc2), nn.ReLU(), nn.Linear(8, 3))
```

When `model` is exported and run through `diff.zono3.interpret`, the
`DiffLinear` output becomes a `DiffExpr2`, which is then promoted to
`DiffExpr3` by the `ReLU` handler.

**Important:** `DiffLinear` is designed for the *input boundary* only.
Chaining two `DiffLinear` layers causes the second one to receive a
`DiffExpr3` input, which is not yet supported.  Use a single `DiffLinear`
at the first layer, then standard layers downstream:

```python
# Correct: one DiffLinear at the start
model = nn.Sequential(
    DiffLinear(fc1a, fc1b),
    nn.ReLU(),
    nn.Linear(hidden, out),   # shared layer — same object
    nn.ReLU(),
    nn.Linear(out, 3),
)

# Not supported: two DiffLinear layers in sequence
# model = nn.Sequential(DiffLinear(a1, a2), nn.ReLU(), DiffLinear(b1, b2))
```

## `diff.zono3.interpret`

`boundlab.diff.zono3.interpret` is an `Interpreter` pre-loaded with all
standard zonotope handlers plus differential handlers for nonlinear ops.

It accepts `DiffExpr3`, `DiffExpr2`, or plain `Expr` inputs:

| Input type | Behaviour |
|---|---|
| `DiffExpr3` | Propagates `(x, y, diff)` through the full differential pipeline. |
| `DiffExpr2` | Promoted to `DiffExpr3` on the first nonlinear layer. |
| `Expr` | Falls back to standard `zono.interpret` (identical bounds). |

```python
import torch
from torch import nn

import boundlab.expr as expr
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.zono3 import interpret as diff_interpret

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
exported = torch.export.export(model, (torch.randn(4),))
op = diff_interpret(exported)

c1, c2 = torch.randn(4), torch.randn(4)
x = expr.ConstVal(c1) + expr.LpEpsilon([4])
y = expr.ConstVal(c2) + expr.LpEpsilon([4])
out = op(DiffExpr3(x, y, x - y))

ub, lb = out.diff.ublb()   # bounds on f(x) - f(y)
```

## ReLU Differential Linearizer

The `relu` differential handler uses a 9-case case split on the sign
patterns of both input branches to derive tight linear relaxations of
`relu(x) - relu(y)`.

Key cases:

| Branch x | Branch y | `relu(x) - relu(y)` |
|---|---|---|
| Dead (`ub ≤ 0`) | Dead | Exactly 0 |
| Active (`lb ≥ 0`) | Active | Exactly `x - y` (pass through) |
| Active | Dead | Bounded by `[lb_x, ub_x]` |
| Dead | Active | Bounded by `[-ub_y, -lb_y]` |
| Crossing | Crossing | VeryDiff relaxation (linear upper/lower) |
| Active/Dead × Crossing | — | Mixed bound |

For the crossing cases a pair of linear functions is fitted that upper-
and lower-bounds `relu(x) - relu(y)` over the joint product of the two
input intervals.

## Registering Custom Differential Linearizers

Use `diff.zono3._register_linearizer` to add differential support for new
nonlinear operations:

```python
from boundlab.diff.zono3 import _register_linearizer
from boundlab.zono import ZonoBounds

@_register_linearizer("my_activation")
def my_diff_linearizer(x, y, d):
    """x, y, d are Expr objects (x = branch1, y = branch2, d = x - y approx)."""
    # Compute standard zonotope bounds for each branch:
    x_bounds = ...   # ZonoBounds for relu(x)
    y_bounds = ...   # ZonoBounds for relu(y)
    # Compute differential bounds; input_weights has three entries [wx, wy, wd]:
    d_bounds = ZonoBounds(
        bias=...,
        error_coeffs=...,
        input_weights=[wx, wy, wd],
    )
    return x_bounds, y_bounds, d_bounds
```

The decorator:

1. Calls your linearizer with the unpacked `(x, y, d)` triple.
2. Builds `DiffExpr3` output expressions from the three `ZonoBounds`.
3. Falls back to the standard `zono.interpret` handler for plain `Expr` inputs.
4. Automatically promotes `DiffExpr2` inputs via `d = x - y`.

## Comparing Differential vs. Independent Bounds

```python
import torch
from torch import nn

import boundlab.expr as expr
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.zono3 import interpret as diff_interpret

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
exported = torch.export.export(model, (torch.randn(4),))

c1 = torch.randn(4)
c2 = c1 + 0.2 * torch.randn(4)
eps = expr.LpEpsilon([4])
x = expr.ConstVal(c1) + eps
y = expr.ConstVal(c2) + eps   # same epsilon — shared noise

# Differential: tracks (x, y, d) jointly
op_diff = diff_interpret(exported)
out = op_diff(DiffExpr3(x, y, x - y))
diff_width = out.diff.bound_width()

# Naive: two independent zonotope runs then subtract
op_std = zono.interpret(exported)
y1 = op_std(expr.ConstVal(c1) + expr.LpEpsilon([4]))
y2 = op_std(expr.ConstVal(c2) + expr.LpEpsilon([4]))
naive_width = (y1.ub() - y2.lb()) - (y1.lb() - y2.ub())

print("Differential width:", diff_width)
print("Naive width:       ", naive_width)
print("Tighter?", (diff_width < naive_width).all().item())  # True
```

## Limitations

- Only `relu` has a differential linearizer.  Other nonlinear ops (`tanh`,
  `exp`, `softmax`) fall back to the standard zonotope handler applied
  independently to each branch, which may be less tight.
- The `diff` component in `DiffExpr2` is not explicitly tracked; call
  `out.x - out.y` to compute it, but be aware this loses the cancellation
  structure that `DiffExpr3.diff` preserves.
- Bound caches (`boundlab.prop._UB_CACHE`, `_LB_CACHE`) are process-global.
  Clear them between experiments if memory is a concern.

## Where To Go Next

- See {doc}`../examples/diff_verification` for a complete differential
  verification script.
- Browse the {doc}`zono_interpret_api` page to understand how standard
  zonotope interpretation works.
- Read `tests/test_diff_zono3.py` for an extensive set of soundness and
  regime-level checks.
