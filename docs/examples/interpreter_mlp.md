# Example: Verify an MLP with `zono.interpret`

This example uses the model interpreter to build an expression graph from an
exported PyTorch program.

```python
import torch
from torch import nn

import boundlab.expr as expr
import boundlab.zono as zono

# Deterministic setup
torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
    nn.ReLU(),
)
model.eval()

x_center = torch.randn(4)
x = expr.ConstVal(x_center) + 0.1 * expr.LpEpsilon([4])

exported = torch.export.export(model, (x_center,))
op = zono.interpret(exported)
y = op(x)

ub, lb = y.ublb()
print("lower:", lb)
print("upper:", ub)

# Optional sanity check by random sampling.
samples = x_center + 0.1 * (torch.rand(2000, 4) * 2 - 1)
with torch.no_grad():
    ys = model(samples)

assert (ys <= ub.unsqueeze(0) + 1e-5).all()
assert (ys >= lb.unsqueeze(0) - 1e-5).all()
```

## What this demonstrates

- Building model-level symbolic transformations from an exported program.
- Running end-to-end bound propagation with `ublb()`.
- A practical Monte-Carlo soundness check.
