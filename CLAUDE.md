# BoundLab

A framework for building neural network verification tools using bound propagation techniques.

## Project Structure

```
src/boundlab/
├── __init__.py          # Package entry point
├── utils.py             # Utility functions (eye_of)
├── expr/                # Expression system for symbolic bound propagation
│   ├── __init__.py      # Exports: Expr, ExprFlags, AffineSum, ConstVal, Add, LpEpsilon, Cat, Stack
│   ├── _core.py         # Expr base class, ExprFlags, expr_pretty_print
│   ├── _affine.py       # AffineSum (fused sum of LinearOp-weighted children), ConstVal
│   ├── _var.py          # LpEpsilon (Lp-norm bounded noise symbol)
│   ├── _cat.py          # Cat, Stack (concatenation/stacking)
│   └── _tuple.py        # TupleExpr, MakeTuple, GetTupleItem (multi-output expressions)
├── linearop/            # Linear operator library for expression backpropagation
│   ├── __init__.py      # Exports all LinearOp types + convenience aliases (NarrowOp, SelectOp, GetItemOp, PadOp)
│   ├── _base.py         # LinearOp base, ComposedOp, SumOp, ScalarOp, ZeroOp
│   ├── _einsum.py       # EinsumOp (general tensor-linear map via Einstein notation)
│   ├── _reshape.py      # ReshapeOp (base), FlattenOp, UnflattenOp, SqueezeOp, UnsqueezeOp (subclasses)
│   ├── _permute.py      # PermuteOp, TransposeOp (unchanged)
│   ├── _expand.py       # ExpandOp (factory → EinsumOp with ones template tensor)
│   ├── _slicing.py      # GetSliceOp, SetSliceOp (structured list[list[slice]] API)
│   ├── _indexing.py     # GetIndicesOp, SetIndicesOp (dim-based index tensor API)
│   ├── _shape.py        # Remaining shape ops: RepeatOp, TileOp, FlipOp, RollOp, DiagOp + re-exports
│   └── _indices.py      # GatherOp, ScatterOp, convenience constructors + re-exports
├── zono/                # Zonotope-based abstract interpretation
│   ├── __init__.py      # ZonoBounds, _register_linearizer, interpret (Interpreter instance)
│   ├── relu.py          # relu_linearizer (triangle relaxation)
│   ├── exp.py           # exp_linearizer
│   ├── reciprocal.py    # reciprocal_linearizer (positive-domain)
│   ├── tanh.py          # tanh_linearizer
│   ├── softmax.py       # softmax_handler (composed from exp + sum + reciprocal)
│   └── bilinear.py      # bilinear_matmul, bilinear_elementwise, matmul_handler
├── interp/              # Abstract interpretation framework
│   └── __init__.py      # Interpreter class, _AFFINE_DISPATCHER
├── poly/                # Polynomial abstract domain (placeholder, not implemented)
│   └── __init__.py
└── prop/                # Propagation utilities
    └── __init__.py      # ub, lb, ublb, center, bound_width (bound computation)
```

## Core Concepts

### Expression System (`boundlab.expr`)
- **Expr**: Base class for all expressions with unique time-ordered ID, shape, children, and backward pass
- **ExprFlags**: Optimization flags (NONE, SYMMETRIC_TO_0, PRINT_FUSE, IS_CONST, IS_AFFINE)
- **backward()**: Computes linear bounds for bound propagation (modes: `>=`, `<=`, `==`)
- **AffineSum**: Fused expression representing `Σ op_i(x_i) + c`; eagerly flattens nested affine sums
- **ConstVal**: Constant tensor values (subclass of AffineSum)
- **Add(*children)**: Convenience function creating an AffineSum with unit coefficients
- **LpEpsilon**: Noise symbol bounded by Lp-norm constraint (`‖ε‖_p ≤ 1`); backward computes `±‖w‖_q` where q is the dual norm
- **Cat, Stack**: Concatenation and stacking of expressions along a dimension
- **TupleExpr, MakeTuple, GetTupleItem**: Multi-output expression support

### Linear Operators (`boundlab.linearop`)
- **LinearOp**: Base class with forward/backward/vforward/vbackward, composition (`@`), addition (`+`)
- **EinsumOp**: General tensor-linear map via Einstein notation
- **ExpandOp**: Factory that returns EinsumOp with ones template tensor; `len(input_shape) == len(output_shape)` enforced (extra dims handled via UnsqueezeOp composition)
- **ComposedOp**: Composition of linear maps (`outer ∘ inner`)
- **SumOp**: Sum of linear maps with matching shapes
- **ScalarOp, ZeroOp**: Scalar multiplication and zero operators
- Reshape ops (all subclasses of ReshapeOp): ReshapeOp, FlattenOp, UnflattenOp, SqueezeOp, UnsqueezeOp
- Permute ops: PermuteOp, TransposeOp (unchanged)
- Slicing ops: GetSliceOp(`input_shape, slices: list[list[slice]]`), SetSliceOp(`output_shape, slices: list[list[slice]]`)
- Indexing ops: GetIndicesOp(`input_shape, dim, indices, added_shape`), SetIndicesOp(`output_shape, dim, indices, added_shape`)
- Other: RepeatOp, TileOp, FlipOp, RollOp, DiagOp, GatherOp, ScatterOp

### Zonotopes (`boundlab.zono`)
Zonotope representation: `Z = c + G ε` where c is center, G is generator matrix, ε ∈ [-1, 1]^m

- **ZonoBounds**: Dataclass with `bias`, `error_coeffs`, `input_weights` fields returned by linearizers
- **_register_linearizer(name)**: Decorator that registers a linearizer function into the `interpret` dispatcher
- **interpret**: Global `Interpreter` instance pre-configured for zonotope abstract interpretation
- Built-in linearizers: relu, exp, reciprocal, tanh, softmax, bilinear matmul

### Abstract Interpretation (`boundlab.interp`)
- **Interpreter**: Dispatches neural network operators to abstract interpretation handlers
  - `__init__(dispatcher, handle_affine=True)`: dispatcher maps operator names to callables; `handle_affine` auto-adds `_AFFINE_DISPATCHER`
  - `__call__(model)`: Takes `nn.Module` or `ExportedProgram`, returns an `interpret(*exprs)` callable
- **_AFFINE_DISPATCHER**: Built-in handlers for arithmetic (`add`, `sub`, `neg`, `mul`, `truediv`, `floordiv`), linear layers (`Linear`, `linear`), and shape ops (`reshape`, `view`, `flatten`, `permute`, `transpose`, `unsqueeze`, `squeeze`, `contiguous`)

### Bound Propagation (`boundlab.prop`)
- **ub(e)**: Compute upper bound using backward-mode propagation
- **lb(e)**: Compute lower bound using backward-mode propagation
- **ublb(e)**: Compute both bounds efficiently (optimizes symmetric expressions)
- **center(e)**: Midpoint of bound interval
- **bound_width(e)**: Width of bound interval (`ub - lb`)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Build docs
pip install -e ".[docs]"
cd docs && make html
```

## Dependencies

- Python ≥3.12
- PyTorch ≥2.0 (supports torch.compile and torch.func)
- sortedcontainers

## Code Style

- Uses Python 3.12+ generic syntax: `class Foo[T: Expr](Expr)`
- Type hints with `Literal[">=", "<=", "=="]` for bound modes
- Expressions are immutable with time-ordered UUIDs for topological sorting
