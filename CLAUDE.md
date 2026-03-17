# BoundLab

A framework for building neural network verification tools using bound propagation techniques.

## Project Structure

```
src/boundlab/
├── __init__.py          # Package entry point
├── utils.py             # Utility functions (eye_of)
├── expr/                # Expression system for symbolic bound propagation
│   ├── __init__.py      # Exports: Expr, ExprFlags, ConstVal, Add, LinearOp, LinearOpSeq, TensorDotLinearOp, LpEpsilon, etc.
│   ├── _core.py         # Expr base class, ExprFlags, expr_pretty_print
│   ├── _base.py         # ConstVal, Add, add
│   ├── _linear.py       # LinearOp, LinearOpSeq, TensorDotLinearOp, contract_linear_ops (VJP-based linear operations)
│   ├── _var.py          # LpEpsilon (Lp-norm bounded noise symbol)
│   └── _cat.py          # Cat, Stack (WIP)
├── zono/                # Zonotope-based verification
│   ├── __init__.py      # ZonoBounds, _register_linearizer, interpret (Interpreter instance)
│   └── relu.py          # relu_linearizer (WIP stub)
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
- **ConstVal**: Constant tensor values
- **Add**: Sum of expressions (use `add()` helper for construction)
- **LinearOp**: Wraps linear functions with VJP support via `torch.func.vjp`
- **LinearOpSeq**: Expression representing a sequence of composed linear operations
- **TensorDotLinearOp**: Specialized LinearOp for tensor dot product transformations
- **LpEpsilon**: Noise symbol bounded by Lp-norm constraint (`‖ε‖_p ≤ 1`); backward computes `±‖w‖_q` where q is the dual norm

### Zonotopes (`boundlab.zono`)
Zonotope representation: `Z = c + G ε` where c is center, G is generator matrix, ε ∈ [-1, 1]^m

- **ZonoBounds**: Dataclass with `bias`, `error_coeffs`, `input_weights` fields returned by linearizers
- **_register_linearizer(name)**: Decorator that registers a linearizer function into the `interpret` dispatcher
- **interpret**: Global `Interpreter` instance pre-configured for zonotope abstract interpretation
- **relu_linearizer**: ReLU linearizer (WIP stub in `zono/relu.py`)

### Abstract Interpretation (`boundlab.interp`)
- **Interpreter**: Dispatches neural network operators to abstract interpretation handlers
  - `__init__(dispatcher, handle_affine=True)`: dispatcher maps operator names to callables; `handle_affine` auto-adds `_AFFINE_DISPATCHER` (add, linear)
  - `__call__(model)`: Takes `nn.Module` or `ExportedProgram`, returns an `interpret(*exprs)` callable (interpretation logic is a WIP stub)
- **_AFFINE_DISPATCHER**: Built-in handlers for `operator.add` and `linear`

### Bound Propagation (`boundlab.prop`)
- **ub(e)**: Compute upper bound using backward-mode propagation
- **lb(e)**: Compute lower bound using backward-mode propagation
- **ublb(e)**: Compute both bounds efficiently (optimizes symmetric expressions)

## Known Issues

- `expr/_var.py:35`: `ExprFlags.NO_DEPENTENCY` is referenced but not defined in `ExprFlags` (typo/missing flag — causes `AttributeError` at runtime)
- `interp/__init__.py`: `Interpreter.__call__` interpretation loop is a stub (`pass`); `poly/` module is empty
- `zono/relu.py`: `relu_linearizer` is unimplemented

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

- Python ≥3.8
- PyTorch ≥2.0 (supports torch.compile and torch.func)
- sortedcontainers

## Code Style

- Uses Python 3.12+ generic syntax: `class Foo[T: Expr](Expr)`
- Type hints with `Literal[">=", "<=", "=="]` for bound modes
- Expressions are immutable with time-ordered UUIDs for topological sorting
