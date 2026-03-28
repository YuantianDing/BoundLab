# BoundLab User Guide

## Mental Model

BoundLab computes sound bounds on neural network outputs under input perturbations.
The workflow has three steps:

1. **Build** a symbolic expression representing the uncertain input.
2. **Transform** it through the network (manually or via `zono.interpret`).
3. **Concretize** the result into numeric upper/lower bounds with `ub`, `lb`, or `ublb`.

```{math}
Z = c + G\epsilon, \quad \epsilon \in [-1,1]^m
```

`Expr` nodes preserve dependency structure so shared symbols are not lost across operations.

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
- `e.ublb()` or `boundlab.prop.ublb(e)`: both at once (more efficient).
- `e.center()`: midpoint of the bound interval.
- `e.bound_width()`: `ub - lb`.

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

## Working Patterns

### Pattern 1: Module-level interpretation (recommended)

Use `zono.interpret(model)` for end-to-end model flows. This is the simplest way to get started.

```python
import torch
from torch import nn
import boundlab.expr as expr
import boundlab.zono as zono

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
x = expr.ConstVal(torch.randn(4)) + 0.1 * expr.LpEpsilon([4])

op = zono.interpret(model)
y = op(x)
ub, lb = y.ublb()
```

### Pattern 2: Manual expression construction

Use this when you need direct control over the symbolic structure.

```python
import torch
import boundlab.expr as expr
import boundlab.zono as zono

x = expr.ConstVal(torch.randn(8)) + expr.LpEpsilon([8])
W = torch.randn(3, 8)
b = torch.randn(3)

y = W @ x + expr.ConstVal(b)
y = zono.interpret.dispatcher["relu"](y)
ub, lb = y.ublb()
```

### Pattern 3: Soundness spot checks with sampling

A practical validation loop:

1. Sample concrete perturbations around your center input.
2. Evaluate the concrete model.
3. Assert all outputs lie in `[lb, ub]`.

This is used extensively in BoundLab tests and is a good regression guard for custom handlers.

## Inspecting Expressions and Operators

BoundLab provides several ways to inspect expressions and linear operators
for debugging and exploration.

### Printing an expression graph

`str(expr)` renders the full expression DAG in SSA (static single assignment)
form. Each node is assigned a `%N` name, and shared subexpressions appear once.
Purely affine sub-graphs are eagerly flattened by `AffineSum`, so the SSA
form is most informative after nonlinear operations (e.g. ReLU).

```python
import torch
import boundlab.expr as expr
import boundlab.zono as zono

x = expr.ConstVal(torch.tensor([1.0, -0.5])) + expr.LpEpsilon([2])
y = zono.interpret.dispatcher["relu"](x)
print(y)
```

```text
bl.Expr {
    %0 = 𝜀_<6>
    %1 = 𝜀_<1>
    %2 = <einsum [2]: [0] -> [0]>(%1) + (set_indices([1] -> [2]) ∘ <einsum [1]: [0] -> [0]>)(%0)
}
```

Here `%1` is the original noise symbol, `%0` is a fresh error symbol
introduced by the ReLU linearizer, and `%2` combines them via EinsumOp
weights.

You can also call `expr.expr_pretty_print(e)` directly to get the SSA body
without the wrapper.

### Key `Expr` attributes

Every expression node exposes:

- `e.shape` — output tensor shape.
- `e.children` — tuple of child expressions.
- `e.flags` — `ExprFlags` (e.g. `IS_AFFINE`, `SYMMETRIC_TO_0`, `IS_CONST`).
- `e.id` — unique time-ordered integer (used for topological ordering).
- `repr(e)` — compact one-line summary: `AffineSum(id=5, flags=ExprFlags.IS_AFFINE)`.

For `AffineSum` nodes specifically:

- `e.children_dict` — `dict[Expr, LinearOp]` mapping each child to its operator.
- `e.constant` — optional constant tensor (`None` if absent).

```python
for child, op in x.children_dict.items():
    print(f"{repr(child)}  ->  {op}  ({op.input_shape} -> {op.output_shape})")
```

### Inspecting `LinearOp`

`str(op)` gives a human-readable description of the operator. Composed
operators display their chain with `∘`:

```python
from boundlab.linearop import ScalarOp, ReshapeOp

a = ScalarOp(2.0, torch.Size([2, 3]))
b = ReshapeOp(torch.Size([2, 3]), (6,))
print(b @ a)  # (reshape([6]) ∘ 2.0)
```

Key attributes:

- `op.input_shape`, `op.output_shape` — tensor shapes.
- `op.flags` — `LinearOpFlags` (e.g. `IS_NON_NEGATIVE`).

You can materialize the full Jacobian for debugging:

- `op.jacobian()` — efficient closed-form Jacobian if the subclass implements
  it (returns `NotImplemented` otherwise).
- `op.force_jacobian()` — always works by applying the operator to a basis;
  returns a tensor of shape `(*output_shape, *input_shape)`.

```python
j = (b @ a).force_jacobian()
print(j.shape)  # torch.Size([6, 2, 3])
```

You can also test operators on concrete tensors:

- `op.forward(x)` — apply the linear map.
- `op.backward(grad)` — apply the transpose.
- `op.vforward(x)` / `op.vbackward(grad)` — batched variants over
  trailing / leading dimensions.

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

## Internals

This section covers implementation details for users who want to understand or extend BoundLab.

### Linear Operators (`boundlab.linearop`)

`boundlab.linearop` provides linear maps used by expression backpropagation and affine rewrites.
Rather than materializing full Jacobian matrices, `LinearOp` stores structured/sparse representations.

Most important types:

- `EinsumOp`: general tensor-linear map (Einstein notation style).
- `ComposedOp`: composition of linear maps (`outer ∘ inner`).
- `SumOp`: sum of linear maps with matching input/output shapes.

Supporting families include:

- Scalar and structural maps (`ScalarOp`, `ZeroOp`).
- Shape transforms (`ReshapeOp`, `PermuteOp`, `FlattenOp`, `TransposeOp`, ...).
- Indexing transforms (`GetSliceOp`, `SetSliceOp`, `GatherOp`, `ScatterOp`, ...).

In bound propagation, each `Expr.backward(...)` returns child weights as `LinearOp` objects. These are composed and merged using `@` and `+`.

### How `ub`, `lb`, and `ublb` work

Concretization uses backward traversal over the expression DAG:

1. Start from identity weight at the query expression.
2. Pop expressions in reverse-topological order.
3. Call local `backward(...)` rules.
4. Accumulate bias terms and push child weights.
5. Finish when all reachable leaves are processed.

`ublb` optimizes this by jointly propagating upper and lower directions and reusing exact paths when available.

Important implementation details:

- Symmetric leaves (for example `LpEpsilon`) can reuse one side by sign flip.
- Shared subexpressions are merged by weight accumulation (dependency-aware).
- Tuple outputs are handled with a dedicated tuple-weight map.
- Results are cached (`_UB_CACHE`, `_LB_CACHE`) for repeated queries.

### `TupleExpr` and multi-output flows

Some traced operations naturally produce multiple outputs. BoundLab supports this via `TupleExpr`-style nodes (for example `MakeTuple` and `GetTupleItem` in `boundlab.expr._tuple`).

Key points:

- `TupleExpr.shape` is a tuple of `torch.Size` objects.
- `TupleExpr.backward(*weights, direction=...)` receives one weight per output slot.
- `GetTupleItem` routes propagation to the parent tuple expression and selected index.

### Linearizers: how nonlinear ops become tractable

Nonlinear functions are handled by linearizers that return a zonotope relaxation (`ZonoBounds`).

A linearizer provides:

- `bias`: constant term.
- `input_weights`: slope-like weights applied to each input expression.
- `error_coeffs`: coefficients for fresh error symbols.

Conceptually:

```{math}
y \approx \sum_i w_i \odot x_i + b + E\eta
```

where `η` is a fresh bounded noise symbol.

Use `@zono._register_linearizer("op_name")` on a function returning `ZonoBounds`.
The decorator installs a dispatcher handler that applies input weights, adds bias, and introduces a fresh `LpEpsilon` for approximation error.

## Limitations and Notes

- BoundLab is currently an early-stage (`0.1.0`) project; interfaces may evolve.
- Unsupported operators in a model trace will raise dispatch errors (typically `KeyError` in interpreter dispatch).
- `softmax` handler currently supports 2D inputs along the last dimension.
- `reciprocal` relaxation assumes positive-domain inputs and clamps lower bounds for numerical safety.
- Some tuple APIs are currently internal (`boundlab.expr._tuple`) and may evolve.
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
