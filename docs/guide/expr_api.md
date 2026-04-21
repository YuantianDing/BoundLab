# API Deep Dive: `Expr`, `backward`, `AffineSum`, `TupleExpr`

This page describes the expression-level APIs used for abstract value construction and bound propagation.

## `Expr`

`Expr` is the core symbolic node type in BoundLab. It represents a relaxed abstract value and forms a DAG with shared subexpressions.

Key properties:

- `id`: unique, time-ordered identifier (used for deterministic traversal).
- `shape`: output tensor shape represented by this node.
- `children`: direct dependency expressions.
- `flags`: optimization/properties metadata (`ExprFlags`).

Key methods:

- `backward(weights, direction)`: local propagation rule.
- `with_children(*new_children)`: rebuild node with different children.
- `ub()`, `lb()`, `ublb()`, `center()`, `bound_width()`: concretization helpers.

## `Expr.backward(...)`

`backward` is the local contract used by global bound propagation.

Signature conceptually:

```python
bias, child_weights = expr.backward(weights, direction)
```

where:

- `weights` is a `LinearOp` encoding an accumulated linear form from output back to this node.
- `direction` is one of:
  - `"=="`: exact affine/equality propagation if available.
  - `"<="`: upper-bound propagation.
  - `">="`: lower-bound propagation.

Return value:

- `bias`: concrete tensor bias contribution to bound.
- `child_weights`: one propagated weight per child.

If a node cannot contribute in a direction (for example a special leaf under exact mode), it may return `None`.

## `AffineSum`

`AffineSum` is the main affine expression implementation.

It represents:

```{math}
\sum_i \mathrm{op}_i(x_i) + c
```

where each `op_i` is a `LinearOp`, `x_i` is a child `Expr`, and `c` is optional constant.

Important behavior:

- Flattens nested affine sums eagerly via operator composition.
- Merges duplicate children by summing their operators.
- Preserves structural sharing and keeps affine graphs compact.

`AffineSum.backward(weights, direction)` is direction-independent (affine exact):

```{math}
\text{bias} = w(c), \quad w_i = w \circ \mathrm{op}_i
```

## `TupleExpr` and `GetTupleItem`

`TupleExpr` models multi-output symbolic values.

Use cases:

- operations producing multiple outputs,
- routing only some outputs downstream,
- preserving dependency alignment for bound propagation.

Key API differences vs `Expr`:

- `TupleExpr.shape` is `tuple[torch.Size, ...]`.
- `TupleExpr.backward(*weights, direction=...)` takes one weight per tuple slot.
- Indexing (`tuple_expr[i]`) yields `GetTupleItem` routing nodes.

`GetTupleItem` is handled specially by propagation code: its incoming weight is redirected to the parent `TupleExpr` at the selected index.

## Minimal Example

```python
import torch
import boundlab.expr as expr
from boundlab.expr._tuple import MakeTuple

x = expr.ConstVal(torch.tensor([1.0, -1.0])) + expr.LpEpsilon([2])
y = 2.0 * x

# Build a symbolic tuple and select outputs.
t = MakeTuple(x, y)
a = t[0]
b = t[1]

ub_a, lb_a = a.ublb()
ub_b, lb_b = b.ublb()
```
