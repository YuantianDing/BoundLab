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
from . import exp as _exp              # registers "exp"
from . import reciprocal as _reciprocal  # registers "reciprocal"
from . import tanh as _tanh            # registers "tanh"

# call_module handlers (mod is the nn.Module instance; pass kwargs when needed)
interpret.dispatcher["ReLU"]      = lambda _, x: interpret.dispatcher["relu"](x)
interpret.dispatcher["Tanh"]      = lambda _, x: interpret.dispatcher["tanh"](x)

# Bilinear matmul handler (supports Expr @ Expr)
from .bilinear import matmul_handler, bilinear_matmul, bilinear_elementwise  # noqa: F401
interpret.dispatcher["matmul"]    = matmul_handler
interpret.dispatcher["bmm"]       = matmul_handler  # alias for batched matmul
interpret.dispatcher["mm"]        = matmul_handler   # alias for 2D matmul

# Softmax handler (composed from exp + sum + reciprocal)
from .softmax import softmax_handler
interpret.dispatcher["softmax"]   = softmax_handler

from .relu import relu_linearizer
from .exp import exp_linearizer
from .reciprocal import reciprocal_linearizer
from .tanh import tanh_linearizer

__all__ = [
    "interpret", "ZonoBounds",
    "relu_linearizer", "exp_linearizer", "reciprocal_linearizer", "tanh_linearizer",
    "bilinear_matmul", "bilinear_elementwise", "matmul_handler", "softmax_handler",  # noqa: F401
]