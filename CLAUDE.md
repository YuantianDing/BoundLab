# BoundLab

A framework for building neural network verification tools using bound propagation techniques.

## Project Structure

```
src/boundlab/
├── __init__.py          # Package entry point
├── expr/                # Expression system for symbolic bound propagation
│   ├── __init__.py      # Expr base class, ExprFlags, pretty printing
│   ├── _base.py         # ConstVal, ConstTensorDot, Add, SubTensor
│   ├── _var.py          # LInfEps (L-infinity perturbation bounds)
│   ├── _mul.py          # ConstTensorDot, ConstMul, ConstMatmul (WIP)
│   └── _utils.py        # einsum_contract_last helper
├── zono/                # Zonotope-based verification
│   ├── __init__.py      # operator() for zonotope transformations
│   └── linearizer.py    # ZonoBounds, linearizers (e.g., ReLU)
└── prop/                # Propagation utilities (WIP)
```

## Core Concepts

### Expression System (`boundlab.expr`)
- **Expr**: Base class for all expressions with unique UUID, shape, children, and backward pass
- **backward()**: Computes linear bounds for bound propagation (modes: `>=`, `<=`, `==`)
- **ConstVal**: Constant tensor values
- **ConstTensorDot**: Linear transformation via tensor dot product
- **Add**: Sum of expressions
- **LInfEps**: L-infinity bounded perturbation term (ε ∈ [-1, 1])

### Zonotopes (`boundlab.zono`)
Zonotope representation: `Z = c + g^T ε` where c is center, g is generators, ε is noise

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
- PyTorch ≥2.0 (supports torch.compile)
- atomicx
- sortedcontainers
- torchtyping

## Code Style

- Uses Python 3.12+ generic syntax: `class Foo[T: Expr](Expr)`
- Type hints with `Literal[">=", "<=", "=="]` for bound modes
- Expressions are immutable with time-ordered UUIDs for topological sorting
