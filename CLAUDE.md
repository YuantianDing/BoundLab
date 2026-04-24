# BoundLab

A framework for building neural network verification tools using bound propagation techniques.

## Project Structure

```
src/boundlab/
├── __init__.py              # Package entry point, lazy-loads submodules
├── utils.py                 # Utility functions (eye_of)
├── expr/                    # Expression system for symbolic bound propagation
│   ├── __init__.py          # Exports: Expr, ExprFlags, AffineSum, ConstVal, Add, LpEpsilon, Cat, Stack
│   ├── _core.py             # Expr base class, ExprFlags, expr_pretty_print
│   ├── _affine.py           # AffineSum (fused sum of LinearOp-weighted children), ConstVal
│   ├── _var.py              # LpEpsilon (Lp-norm bounded noise symbol)
│   ├── _cat.py              # Cat, Stack (concatenation/stacking)
│   └── _tuple.py            # TupleExpr, MakeTuple, GetTupleItem (multi-output expressions)
├── linearop/                # Linear operator library for expression backpropagation
│   ├── __init__.py          # Exports all LinearOp types + convenience aliases
│   ├── _base.py             # LinearOp base, ComposedOp, SumOp, ScalarOp, ZeroOp
│   ├── _einsum.py           # EinsumOp (general tensor-linear map via Einstein notation)
│   ├── _reshape.py          # ReshapeOp (base), FlattenOp, UnflattenOp, SqueezeOp, UnsqueezeOp
│   ├── _permute.py          # PermuteOp, TransposeOp
│   ├── _expand.py           # ExpandOp (factory → EinsumOp with ones template tensor)
│   ├── _slicing.py          # GetSliceOp, SetSliceOp (structured list[list[slice]] API)
│   ├── _indexing.py         # GetIndicesOp, SetIndicesOp (dim-based index tensor API)
│   ├── _shape.py            # RepeatOp, TileOp, FlipOp, RollOp, DiagOp + re-exports
│   └── _indices.py          # GatherOp, ScatterOp, convenience constructors + re-exports
├── zono/                    # Zonotope-based abstract interpretation (v2 - stable API)
│   ├── __init__.py          # ZonoBounds, _register_linearizer, interpret (Interpreter instance)
│   ├── relu.py              # relu_linearizer (triangle relaxation)
│   ├── exp.py               # exp_linearizer
│   ├── reciprocal.py        # reciprocal_linearizer (positive-domain)
│   ├── tanh.py              # tanh_linearizer
│   ├── softmax.py           # softmax_handler (composed from exp + sum + reciprocal)
│   └── bilinear.py          # bilinear_matmul, bilinear_elementwise, matmul_handler
├── diff/                    # Differential verification toolkit
│   ├── __init__.py          # Exports: zono3, op, net
│   ├── expr.py              # Expression utilities for differential verification
│   ├── net.py               # Network utilities for differential verification
│   ├── op.py                # Operator definitions for differential verification
│   ├── certify_split.py     # Split certification algorithm
│   ├── delta_top1.py        # Top-1 delta verification
│   └── zono3/               # Advanced zonotope abstract interpretation (v3 - experimental)
│       ├── __init__.py      # Enhanced interpret dispatcher
│       ├── default/         # Default linearizers (relu, exp, tanh, reciprocal, softmax, heaviside)
│       ├── gradlin/         # Gradient-based linearizers
│       ├── _hex_cheby.py    # Chebyshev polynomial utilities
│       ├── bilinear.py, exp.py, relu.py, tanh.py, reciprocal.py, softmax.py, heaviside.py
│       └── zonohex/         # Hexagon zonotope variant (experimental)
├── gradlin/                 # Gradient-descent-based linear bound tightening
│   ├── __init__.py          # gradlin, trapezoid_region
│   └── _core.py             # Core gradient-based linearization algorithm
├── interp/                  # Abstract interpretation framework
│   ├── __init__.py          # Interpreter class, _AFFINE_DISPATCHER, ONNX_BASE_INTERPRETER
│   └── onnx.py              # ONNX export utilities, onnx_export, register_onnx_translation
├── sparse/                  # Sparse tensor operations (experimental)
│   ├── __init__.py          # Module exports
│   ├── coos.py              # Sparse COOS format utilities
│   ├── factors.py           # Sparse factorizations
│   ├── ops.py               # Sparse operations
│   └── table.py             # Sparse table structures
├── poly/                    # Polynomial abstract domain (placeholder, not yet implemented)
│   └── __init__.py
└── prop/                    # Propagation utilities
    └── __init__.py          # ub, lb, ublb, center, bound_width (bound computation)
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

### Differential Verification (`boundlab.diff`)
Module for verifying differential properties of neural networks. Contains experimental zonotope v3 with enhanced linearizers:
- **zono3**: Advanced zonotope interpretation with two linearizer variants:
  - **default/**: Standard linearizers for relu, exp, reciprocal, tanh, softmax, heaviside
  - **gradlin/**: Gradient-based linearizers for tighter bounds
- **certify_split**: Splitting-based certification algorithm
- **delta_top1**: Top-1 difference verification
- **net**: Network utilities for differential verification
- **op**: Operator abstractions for differential verification

### Gradient-based Linearization (`boundlab.gradlin`)
Finds tight linear bounds for smooth functions over polytope regions:
- **gradlin(f, grad_inv, lb, ub, A, b, iters)**: Find linear bound `lam·x + L ≤ f(x) ≤ lam·x + U`
- **trapezoid_region(lx, ux, ly, uy, ld, ud)**: Construct polytope constraints for trapezoid regions
Used internally by zono3 gradlin linearizers for improved bound tightness.

### ONNX Support (`boundlab.interp.onnx`)
Utilities for converting PyTorch modules to ONNX IR for abstract interpretation:
- **onnx_export(f, input_shapes)**: Export a callable or nn.Module to ONNX IR model
- **register_onnx_translation(op_name)**: Register custom torch op → ONNX translation
- **ONNX_BASE_INTERPRETER**: Pre-configured interpreter for common ONNX ops
Custom ops use sentinel functions during export, later replaced with primitive nodes.

### Sparse Operations (`boundlab.sparse`)
Experimental module for sparse tensor abstractions (under development):
- **coos**: Coordinate format sparse tensor handling
- **factors**: Sparse matrix factorizations
- **ops**: Sparse tensor operations
- **table**: Sparse table data structures

## Development

### Installation

```bash
# Basic installation (from source)
pip install -e .

# With dev dependencies (testing, coverage)
pip install -e ".[dev]"

# With docs dependencies
pip install -e ".[docs]"
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basics.py

# Run with coverage
pytest --cov=src/boundlab tests/

# Run specific test (useful for debugging)
pytest tests/test_zono.py::test_something -vv
```

Key test files:
- `tests/test_basics.py`: Core expr and linearop functionality
- `tests/test_zono.py`: Zonotope interpretation
- `tests/test_bert_verification.py`: Large model verification (BERT)
- `tests/test_vit.py`: Vision transformer verification
- `tests/test_heaviside_linearizer.py`: Heaviside function linearization
- `tests/test_certify_split.py`: Differential certification

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs && make html
# Output in docs/_build/html/
```

### Running Examples

```bash
# Examples in examples/ directory
python examples/certify_vit.py
python examples/certify_transformer.py

# Experiments in experiments/ directory
python experiments/certify_notanh.py
```

## Dependencies

- **Core**: Python ≥3.8, PyTorch ≥2.0, sortedcontainers
- **ONNX support**: onnx_ir, optional but needed for onnx_export
- **Dev**: pytest ≥7.0, pytest-cov ≥4.0
- **Docs**: sphinx ≥7.0, pydata-sphinx-theme, sphinx-design, myst-parser

## Code Style & Conventions

### Python & Type System
- Uses Python 3.8+ (targets 3.12 for advanced features like `class Foo[T: Expr](Expr)`)
- Type hints with `Literal[">=", "<=", "=="]` for bound modes
- All public functions have type signatures
- Import order: stdlib, third-party (torch), local

### Expression Design
- **Immutability**: Expr objects are immutable; time-ordered UUIDs enable topological sorting
- **Backward Pass**: Implement `backward(g, mode)` to propagate bounds symbolically
- **Lazy Evaluation**: Use `backward()` not direct computation; enables optimization
- **No Side Effects**: Methods like `ub()`, `lb()` should be pure functions

### LinearOp Design
- **Composition**: Use `@` operator for composition (`outer @ inner`)
- **Vectorized Operations**: Implement `vforward` and `vbackward` for batch processing
- **Shape Contracts**: Always document input/output shapes in docstrings
- **Reusability**: Prefer factory functions (e.g., ExpandOp) over direct instantiation

### Zonotope Linearizers
- **Registration**: Use `@_register_linearizer("op_name")` decorator
- **Return Type**: Always return `ZonoBounds` dataclass
- **Bounds Tightness**: Prefer tighter over looser bounds (enables certification)
- **Experimental Features**: zono3 with gradlin are experimental; stable zono/ is v2

### Testing Patterns
- **Tensor Equality**: Use `torch.allclose()` for numerical stability
- **Shape Verification**: Test input/output shapes explicitly
- **Type Safety**: Verify type conversions (e.g., int → tensor)
- **Boundary Cases**: Test at domain boundaries (e.g., ReLU at x=0)

## Common Patterns & Workflows

### Adding a New Linearizer
1. Create `src/boundlab/zono/my_op.py` with `my_op_linearizer(x: Expr, ...) -> ZonoBounds`
2. Register with `@_register_linearizer("my_op")`
3. Test against reference implementation (e.g., `torch.my_op`)
4. Add test in `tests/test_zono.py` with various domain values
5. Document bounds tightness and any assumptions (e.g., domain restrictions)

### Verifying a Neural Network
```python
from boundlab.expr import ConstVal, LpEpsilon
from boundlab.zono import interpret

# 1. Create input distribution
x_center = ConstVal(torch.randn(1, 3, 224, 224))
x_noise = LpEpsilon([1, 3, 224, 224], p=float('inf'))
x = x_center + x_noise * epsilon  # epsilon ≤ eps_bound

# 2. Build interpreter for your ops
interp = interpret  # Use default zonotope v2

# 3. Run forward pass
output = interp(model)(x)

# 4. Compute bounds
from boundlab.prop import ub, lb
upper = ub(output)
lower = lb(output)
```

### Extending the Interpreter
```python
from boundlab.interp import Interpreter

my_dispatcher = {
    "custom_op": lambda x: x,  # Handle identity
    "my_nonlinearity": my_linearizer_func,
}
my_interp = Interpreter(my_dispatcher, handle_affine=True)

# Now use: my_interp(model)(x)
```

## Architecture & Design Rationale

### Why Immutable Expressions?
Enables static analysis, memoization, and topological sorting for correct bound propagation order.

### Why Backward Mode for Bounds?
Backward-mode propagation allows computing bounds on final layer outputs (verification goal) without materializing intermediate tensor values.

### Why Zonotope Abstract Domain?
Affine arithmetic + error intervals provide tight bounds while remaining computationally efficient. Linear ops compose naturally.

### Why Separate expr, linearop, zono?
- **expr**: Symbolic computation graph (language-level)
- **linearop**: Linear algebra (algorithm-level)
- **zono**: Concrete abstract interpretation (domain-level)
Clear separation of concerns enables mixing domains or linearization strategies.

## Experimental Features & Stability

- **zono/ (v2)**: Stable, well-tested, recommended for production
- **diff/zono3/ (v3)**: Experimental with enhanced linearizers; API may change
- **gradlin/**: Experimental; used internally by zono3/gradlin
- **sparse/**: Not yet production-ready; placeholder for future work
- **poly/**: Not implemented

## AI Assistant Guidelines

When working on this codebase:

1. **Understand the Expression Graph**: Before implementing features, trace through how expressions flow: model → expr → backward passes → bounds
2. **Test Against PyTorch Semantics**: Verify linearizers match `torch.relu()`, `torch.exp()`, etc. on random inputs
3. **Respect Immutability**: Don't modify Expr objects post-creation; create new ones instead
4. **Compose Liberally**: Use LinearOp composition and Expr Add() to build complex operations
5. **Document Assumptions**: Zonotope linearizers often have domain restrictions (e.g., reciprocal on positive domain) — document these
6. **Use Existing Patterns**: Copy structure from similar linearizers rather than inventing new approaches
7. **Consider Both Bounds**: When testing, verify upper AND lower bounds; asymmetric functions need both
8. **Profile Large Models**: Use examples/ scripts for integration tests (BERT, ViT are good sanity checks)
