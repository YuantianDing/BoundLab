# API Deep Dive: `zono.interpret` and Linearizers

This page explains how zonotope interpretation works for neural network architectures.

## `zono.interpret(program_or_graph_module)`

`boundlab.zono.interpret` is an `Interpreter` configured with zonotope handlers.
It accepts exported IR (`torch.export.ExportedProgram`) or
`torch.fx.GraphModule` inputs, not raw `nn.Module`.

Usage:

```python
import boundlab.zono as zono

op = zono.interpret(program_or_graph_module)
y_expr = op(x_expr)
ub, lb = y_expr.ublb()
```

Workflow:

1. Prepare exported/FX graph IR from your model.
2. Dispatch each node by op name/module/method.
3. Build output expression(s) in the zonotope abstract domain.

## Linearizer Contract

Nonlinear operations are implemented via linearizers.
Each linearizer takes one or more `Expr` inputs and returns `ZonoBounds`:

- `bias`: constant term.
- `input_weights`: linear weights applied to each input `Expr`.
- `error_coeffs`: coefficients of fresh approximation noise symbols.

Conceptually, for one-input ops:

```{math}
y \approx w \odot x + b + E\eta
```

where `η` is a fresh bounded error symbol.

## Registration and Execution

Register a linearizer with:

```python
@zono._register_linearizer("op_name")
def my_linearizer(x):
    return ZonoBounds(...)
```

The registration wrapper builds the output expression by:

1. summing weighted inputs (`input_weights`),
2. adding `bias`,
3. creating fresh `LpEpsilon` noise,
4. applying `error_coeffs` to the new noise symbol.

This turns local relaxations into composable expression-level transformers.

## Built-in Zonotope Handlers

BoundLab currently includes handlers for:

- `relu`
- `exp`
- `reciprocal`
- `tanh`
- `softmax` (composed from `exp`, sum, reciprocal, element-wise product)
- bilinear `matmul` relaxations for `Expr @ Expr`

## Minimal Example

```python
import torch
from torch import nn

import boundlab.expr as expr
import boundlab.zono as zono

model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
x = expr.ConstVal(torch.randn(4)) + 0.1 * expr.LpEpsilon([4])

exported = torch.export.export(model, (torch.randn(4),))
op = zono.interpret(exported)
y = op(x)
ub, lb = y.ublb()
```

## Notes

- Unsupported traced operators will fail dispatch (usually a `KeyError`).
- Softmax support is currently focused on 2D inputs along the last dimension.
- Numerical stability for softmax is handled via center-shifting before `exp`.
