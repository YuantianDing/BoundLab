# BoundLab User Guide

## Mental Model

`Expr` represents a relaxed abstract value of a variable.
For example, a zonotope can be encoded as an expression that symbolically captures center terms, linear dependencies, and error terms.

Transformations are applied directly to these expressions.
For example, `zono.interpret(model)` runs abstract interpretation over a model and returns a transformed `Expr` describing the output abstract value.

The abstract result is then concretized into numeric bounds.
Methods such as dual-norm concretization and backward bound propagation (in the spirit of tools like auto_LiRPA) are used to compute concrete `ub`, `lb`, and `ublb`.

## Core Building Blocks

### Expressions (`boundlab.expr`)

The central abstraction is `Expr`. Most workflows use:

- `ConstVal(tensor)`: deterministic value.
- `LpEpsilon(shape, p="inf")`: perturbation symbol with bounded norm.
- Affine expression composition using operators like `+`, `-`, `*`, `@`, reshape/index ops.
- `Cat` and `Stack` for concatenation/stacking symbolic tensors.

Typical uncertain input construction:

```python
import torch
import boundlab.expr as expr

x_center = torch.randn(8)
x = expr.ConstVal(x_center) + 0.1 * expr.LpEpsilon([8])
```

### Bound Propagation (`boundlab.prop`)

Given any expression `e`, BoundLab can compute:

- `e.ub()` or `boundlab.prop.ub(e)`: upper bound.
- `e.lb()` or `boundlab.prop.lb(e)`: lower bound.
- `e.ublb()` or `boundlab.prop.ublb(e)`: both at once.
- `e.center()`: midpoint of the bound interval.
- `e.bound_width()`: `ub - lb`.

Example:

```python
ub, lb = x.ublb()
print("max width:", (ub - lb).max().item())
```

### Zonotope Interpreter (`boundlab.zono.interpret`)

`zono.interpret(model)` returns a callable that maps input expressions to output expressions.

```python
from torch import nn
import boundlab.zono as zono

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
op = zono.interpret(model)
y = op(x)
ub, lb = y.ublb()
```

Under the hood, BoundLab traces your model using `torch.fx` and dispatches each operation to registered handlers.

## Supported Operations

### Affine and shape operations

BoundLab includes handlers for many affine operations, including:

- Arithmetic: `add`, `sub`, `neg`, scalar/tensor `mul`, division.
- Linear layers: `nn.Linear`, `torch.nn.functional.linear`.
- Shape transforms: `reshape`, `view`, `flatten`, `permute`, `transpose`, `unsqueeze`, `squeeze`, `contiguous`.
- Indexing and scatter/gather primitives through expression methods.

### Nonlinear zonotope linearizers

The zonotope domain currently registers linearizers/handlers for:

- `relu`
- `exp`
- `reciprocal` (for positive-domain inputs)
- `tanh`
- `softmax` (implemented as composition of `exp`, sum, reciprocal, element-wise product)
- bilinear `matmul` (McCormick-style relaxation when both operands are symbolic)

## Working Patterns

### Pattern 1: Manual expression construction

Use this when you need direct control over symbolic structure.

```python
W = torch.randn(3, 8)
b = torch.randn(3)

y = W @ x + expr.ConstVal(b)
y = zono.interpret.dispatcher["relu"](y)
ub, lb = y.ublb()
```

### Pattern 2: Module-level interpretation

Use `zono.interpret(model)` for end-to-end model flows.

```python
op = zono.interpret(model)
out_expr = op(x)
ub, lb = out_expr.ublb()
```

### Pattern 3: Soundness spot checks with sampling

A practical validation loop:

1. Sample concrete perturbations around your center input.
2. Evaluate the concrete model.
3. Assert all outputs lie in `[lb, ub]`.

This is used extensively in BoundLab tests and is a good regression guard for custom handlers.

## Limitations and Notes

- BoundLab is currently an early-stage (`0.1.0`) project; interfaces may evolve.
- Unsupported operators in a model trace will raise dispatch errors (typically `KeyError` in interpreter dispatch).
- `softmax` handler currently supports 2D inputs along the last dimension.
- `reciprocal` relaxation assumes positive-domain inputs and clamps lower bounds for numerical safety.
- For repeated experiments in one process, you may want to clear propagation caches:

```python
import boundlab.prop as prop
prop._UB_CACHE.clear()
prop._LB_CACHE.clear()
```

## Extending BoundLab

To add a new nonlinear operator in the zonotope domain:

1. Implement a linearizer returning `zono.ZonoBounds`.
2. Register it with `@zono._register_linearizer("op_name")`.
3. Add dispatcher entries for both function and module forms if needed.
4. Add soundness tests (sample-based checks are a good baseline).

## Where To Go Next

- See {doc}`../examples/index` for complete scripts.
- Inspect {doc}`../api/index` for full signatures and class docs.
- Read `tests/` for additional usage patterns and soundness checks.
