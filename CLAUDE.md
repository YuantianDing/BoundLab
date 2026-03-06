# BoundLab

A framework for building neural network verification tools using bound propagation techniques.

## Project Structure

```
src/boundlab/
├── __init__.py          # Package entry point
├── utils.py             # Utility functions (eye_of)
├── expr/                # Expression system for symbolic bound propagation
│   ├── __init__.py      # Exports: Expr, ExprFlags, ConstVal, Add, LinearOp, LinearOpSeq, TensorDotLinearOp, etc.
│   ├── _core.py         # Expr base class, ExprFlags, expr_pretty_print
│   ├── _base.py         # ConstVal, Add, add
│   ├── _linear.py       # LinearOp, LinearOpSeq, linear_op (VJP-based linear operations)
│   └── _cat.py          # Cat, Stack (WIP)
├── zono/                # Zonotope-based verification
│   ├── __init__.py      # operator() for zonotope transformations
│   └── linearizer.py    # ZonoBounds, ZonoLinearizer, relu_linearizer, DEFAULT_LINEARIZER
└── prop/                # Propagation utilities
    └── __init__.py      # ub, lb, ublb (bound computation)
```

## Core Concepts

### Expression System (`boundlab.expr`)
- **Expr**: Base class for all expressions with unique time-ordered ID, shape, children, and backward pass
- **ExprFlags**: Optimization flags (NONE, SYMMETRIC_TO_0, PRINT_FUSE, IS_CONST)
- **backward()**: Computes linear bounds for bound propagation (modes: `>=`, `<=`, `==`)
- **ConstVal**: Constant tensor values
- **Add**: Sum of expressions (use `add()` helper for construction)
- **LinearOp**: Wraps linear functions with VJP support via `torch.func.vjp`
- **LinearOpSeq**: Expression representing a sequence of composed linear operations
- **TensorDotLinearOp**: Specialized LinearOp for tensor dot product transformations

### Zonotopes (`boundlab.zono`)
Zonotope representation: `Z = c + G ε` where c is center, G is generator matrix, ε ∈ [-1, 1]^m

### Bound Propagation (`boundlab.prop`)
- **ub(e)**: Compute upper bound using backward-mode propagation
- **lb(e)**: Compute lower bound using backward-mode propagation
- **ublb(e)**: Compute both bounds efficiently (optimizes symmetric expressions)

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
