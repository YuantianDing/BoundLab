# API Deep Dive: `LinearOp`

`LinearOp` is the operator abstraction behind affine propagation in BoundLab.

## Why `LinearOp` Exists

Any linear tensor transformation can be written as multiplication by a Jacobian-like tensor:

```{math}
y = Jx, \quad J \in \mathbb{R}^{[*\,\text{output\_shape}, *\,\text{input\_shape}]}
```

Materializing `J` explicitly is usually expensive. `LinearOp` stores a structured/sparse representation and supports efficient algebra (`forward`, transpose action, composition, addition).

## Core API

Base class: `boundlab.linearop.LinearOp`

Primary fields:

- `input_shape`
- `output_shape`
- `flags` (`LinearOpFlags`, e.g. non-negativity metadata)

Primary methods:

- `forward(x)`: apply linear map.
- `backward(grad_output)`: apply transposed linear map.
- `vforward(x)`: batched forward over trailing batch dims.
- `vbackward(grad_output)`: batched backward over leading batch dims.
- `jacobian()`: optional explicit Jacobian (if provided efficiently).
- `force_jacobian()`: explicit Jacobian fallback via batched basis application.
- `jacobian_scatter(src)`: add this operator's Jacobian into an existing
  Jacobian-layout tensor without requiring a full re-materialization path.
- `abs()`, `sum_input()`, `sum_output()`: specialized algebra helpers.

Operator overloads:

- `A @ B`: composition (`ComposedOp`).
- `A + B`: sum (`SumOp`).
- `s * A`: scalar scaling (`ScalarOp` composition).

## Most Important Implementations

### `EinsumOp`

General tensor-linear map using Einstein-index semantics. This is the workhorse for dense affine transforms, including matrix multiplication and elementwise weighted maps.

### `ComposedOp`

Represents functional composition of linear maps:

```{math}
(\mathrm{outer} \circ \mathrm{inner})(x) = \mathrm{outer}(\mathrm{inner}(x))
```

Used when propagation chains through multiple affine steps.

### `SumOp`

Represents pointwise sum of compatible linear maps:

```{math}
(A + B)(x) = A(x) + B(x)
```

Used heavily when multiple symbolic branches contribute to the same child.

## Shape and Indexing LinearOps

BoundLab also provides linear operators for structural tensor ops:

- shape ops (`ReshapeOp`, `PermuteOp`, `TransposeOp`, ...)
- indexing/slicing ops (`GetSliceOp`, `SetSliceOp`, `GetIndicesOp`, `SetIndicesOp`)
- gather/scatter ops (`GatherOp`, `ScatterOp`)

These stay inside the same `LinearOp` algebra, so propagation does not need separate code paths.

## Jacobian Utility APIs

Two helper APIs are useful for debugging and advanced composition logic:

- `force_jacobian()`: always returns a dense tensor of shape
  `[*output_shape, *input_shape]` (potentially expensive).
- `jacobian_scatter(src)`: returns `src + J` where `J` is this operator's
  Jacobian. Subclasses like `EinsumOp` can implement this with structured
  scatter logic to avoid unnecessary full-Jacobian construction.

## Minimal Example

```python
import torch
from boundlab.linearop import ScalarOp, ReshapeOp

x = torch.randn(2, 3)
a = ScalarOp(2.0, torch.Size([2, 3]))
b = ReshapeOp(torch.Size([2, 3]), (6,))

op = b @ a      # compose: scale then reshape
y = op.forward(x)

g = torch.randn(6)
dx = op.backward(g)
```
