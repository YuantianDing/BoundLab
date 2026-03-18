r"""Zonotope-Based Abstract Interpretation for Neural Networks

This module provides zonotope transformations for computing over-approximations
of neural network outputs under bounded input perturbations.
"""


import dataclasses
import torch

from boundlab import expr
from boundlab.expr._core import Expr
from boundlab.expr._var import LpEpsilon
from boundlab.interp import Interpreter
from boundlab.linearop import LinearOp

interpret = Interpreter({})
"""Zonotope-Based Abstract Interpretation for Neural Networks"""

@dataclasses.dataclass
class ZonoBounds:
    """Data class representing zonotope bounds for a neural network layer."""
    bias: torch.Tensor # The bias term of the zonotope
    error_coeffs: LinearOp
    input_weights: list[torch.Tensor]  # Hadamard product weights of the input terms


def _register_linearizer(name: str):
    def decorator(linearizer: callable):
        def handler(*exprs: Expr) -> Expr:
            bounds = linearizer(*exprs)
            assert all(w.shape == e.shape for w, e in zip(bounds.input_weights, exprs)), \
                "Input weights must match the shapes of the input expressions."
            # Apply slopes to input expressions
            result_expr = sum(w * e for w, e in zip(bounds.input_weights, exprs)) + bounds.bias
            # Introduce a fresh noise symbol for the approximation error
            new_eps = LpEpsilon(bounds.error_coeffs.input_shape)
            result_expr = result_expr + bounds.error_coeffs(new_eps)
            return result_expr
        interpret.dispatcher[name] = handler
        return linearizer
    return decorator


# Import activation modules last so _register_linearizer and ZonoBounds are already defined
from . import relu as _relu            # registers "relu"

# call_module handlers (mod is the nn.Module instance; pass kwargs when needed)
interpret.dispatcher["ReLU"]      = lambda _, x: interpret.dispatcher["relu"](x)

__all__ = ["interpret", "ZonoBounds"]