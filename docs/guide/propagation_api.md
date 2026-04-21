# API Deep Dive: `ub`, `lb`, `ublb`

This page explains concretization APIs in `boundlab.prop`.

## `ub` and `lb`

- `ub(e)`: compute an upper bound tensor for expression `e`.
- `lb(e)`: compute a lower bound tensor for expression `e`.

They perform backward bound propagation over the expression DAG, similar in spirit to LiRPA-style backward propagation.

High-level behavior:

1. Initialize root weight with identity map.
2. Traverse nodes in reverse-topological order.
3. Call each node's `backward(...)` rule with direction `"<="` (for `ub`) or `">="` (for `lb`).
4. Accumulate bias terms and propagate child weights.

## `ublb`

`ublb(e)` computes both bounds in one pass where possible and returns:

```python
ub_tensor, lb_tensor = ublb(e)
```

Why it is useful:

- Reduces duplicated traversal work.
- Reuses exact affine propagation paths when available.
- Exploits symmetry for symmetric abstract leaves (e.g. zonotope noise symbols), which is especially useful for zonotope-like domains.

Conceptually:

```{math}
\mathrm{width}(e) = \mathrm{ub}(e) - \mathrm{lb}(e)
```

and `ublb` aims to produce both terms with shared computation.

## Caching and Practical Notes

Propagation results are cached by expression id in internal caches (`_UB_CACHE`, `_LB_CACHE`).

This helps repeated concretization calls, but for repeated experimental runs in one process you may want to clear caches:

```python
import boundlab.prop as prop
prop._UB_CACHE.clear()
prop._LB_CACHE.clear()
```

## Minimal Example

```python
import torch
import boundlab.expr as expr

x = expr.ConstVal(torch.tensor([0.5, -1.0])) + expr.LpEpsilon([2])

ub = x.ub()
lb = x.lb()
ub2, lb2 = x.ublb()

assert torch.allclose(ub, ub2)
assert torch.allclose(lb, lb2)
```
