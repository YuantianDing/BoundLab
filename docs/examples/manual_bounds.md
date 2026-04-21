# Example: Manual Bounds on an Affine + ReLU Graph

This example builds the expression graph manually and applies the registered ReLU linearizer.

```python
import torch

import boundlab.expr as expr
import boundlab.zono as zono

# x in center + [-1, 1]^4
center = torch.tensor([0.2, -0.5, 1.0, -1.2])
x = expr.ConstVal(center) + expr.LpEpsilon([4])

# One affine layer y = W x + b
W = torch.tensor([
    [1.0, -0.2, 0.4, 0.0],
    [-0.5, 0.8, 0.0, 1.1],
])
b = torch.tensor([0.1, -0.3])

y_lin = W @ x + expr.ConstVal(b)

# Apply zonotope ReLU handler directly.
relu = zono.interpret["relu"]
y = relu(y_lin)

ub, lb = y.ublb()
print("lb =", lb)
print("ub =", ub)
print("width =", ub - lb)
```

## What this demonstrates

- How to model uncertainty with `ConstVal + LpEpsilon`.
- How to compose affine maps with `@` and `+`.
- How to invoke nonlinear handlers from the zonotope dispatcher.
- How to concretize a final expression via `ublb()`.
