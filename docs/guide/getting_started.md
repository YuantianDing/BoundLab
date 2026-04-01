# Getting Started

This page walks through installation and a minimal end-to-end bound computation.

Core workflow: `Expr` (relaxed abstract value) -> transform (`zono.interpret` or expression ops) -> dual-norm concretization (`ub` / `lb` / `ublb`).

## Install

### From source (recommended for development)

```bash
git clone https://github.com/YuantianDing/boundlab.git
cd boundlab
pip install -e .
```

### Install docs dependencies (optional)

```bash
pip install -e ".[docs]"
```

### Verify installation

```bash
python -c "import boundlab as bl; print(bl.__version__)"
```

## Quick Start: Bound a Simple ReLU Network

```python
import torch
from torch import nn

import boundlab.expr as expr
import boundlab.zono as zono

# A small model we want to verify under input perturbation.
model = nn.Sequential(
    nn.Linear(4, 6),
    nn.ReLU(),
    nn.Linear(6, 3),
)

# Nominal input center.
x_center = torch.randn(4)

# Build symbolic input: x = center + eps, where eps_i in [-1, 1].
x_expr = expr.ConstVal(x_center) + expr.LpEpsilon([4])

# Export first, then build an abstract interpreter and propagate bounds.
exported = torch.export.export(model, (x_center,))
op = zono.interpret(exported)
y_expr = op(x_expr)

ub, lb = y_expr.ublb()
print("Upper bound:", ub)
print("Lower bound:", lb)
print("Width:", ub - lb)
```

## Next Steps

- Continue with the {doc}`index` to learn the expression system and interpreter internals.
- Explore runnable patterns in {doc}`../examples/index`.
- Browse full APIs in {doc}`../api/index`.

## Build This Documentation Locally

```bash
cd docs
make html
```

The generated site is written to `docs/_build/html`.
