r"""Zonotope-Based Abstract Interpretation for Neural Networks

This module provides zonotope transformations for computing over-approximations
of neural network outputs under bounded input perturbations.
"""


import dataclasses
import torch

from boundlab import expr
from boundlab.expr._core import Expr
from boundlab.expr._linear import contract_linear_ops
from boundlab.interp import Interpreter

from .relu import relu_linearizer

interpret = Interpreter({})
"""Zonotope-Based Abstract Interpretation for Neural Networks"""

@dataclasses.dataclass
class ZonoBounds:
    """Data class representing zonotope bounds for a neural network layer."""
    bias: torch.Tensor # The bias term of the zonotope
    error_coeffs: torch.Tensor # Hardmard product weights of the error terms
    input_weights: list[torch.Tensor] # Hardmard product weights of the input terms


def _register_linearizer(name: str):
    def decorator(linearizer: callable):
        def handler(*exprs: Expr) -> Expr:
            bounds = linearizer(*exprs)
            assert all(w.shape == e.shape for w, e in zip(bounds.input_weights, exprs)), "Input weights must match the shapes of the input expressions."
            result_expr = sum(w * e for w, e in zip(bounds.input_weights, bounds.error_coeffs)) + bounds.bias
            return contract_linear_ops(result_expr)
        interpret.dispatcher[name] = handler
        return linearizer
    return decorator

from . import relu

# TODO: Add more linearizers for other nonlinear activations, such as sigmoid, tanh, etc., and register them with the interpreter.