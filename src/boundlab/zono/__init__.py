r"""Zonotope-Based Abstract Interpretation for Neural Networks

This module provides zonotope transformations for computing over-approximations
of neural network outputs under bounded input perturbations.

Examples
--------
>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> import boundlab.zono as zono
>>> model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
>>> op = zono.interpret(model)
>>> x = expr.ConstVal(torch.zeros(4)) + expr.LpEpsilon([4])
>>> y = op(x)
>>> y.ub().shape
torch.Size([3])
"""

from __future__ import annotations

import dataclasses
import torch

from boundlab import expr
from boundlab.expr._core import Expr
from boundlab.expr._var import LpEpsilon
from boundlab.interp import Interpreter
from boundlab.linearop import LinearOp

interpret = Interpreter({})
"""Zonotope-based interpreter.

Examples
--------
>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> import boundlab.zono as zono
>>> op = zono.interpret(nn.Linear(2, 1))
>>> y = op(expr.ConstVal(torch.zeros(2)) + expr.LpEpsilon([2]))
>>> y.shape
torch.Size([1])
"""


@dataclasses.dataclass
class ZonoBounds:
    """Data class representing zonotope bounds for a neural network layer.

    Examples
    --------
    ``input_weights`` has one entry per input expression to the linearizer.
    For unary ops such as ReLU, this is typically a single slope tensor.
    """
    bias: torch.Tensor # The bias term of the zonotope
    error_coeffs: LinearOp
    input_weights: list[torch.Tensor | 0]  # Hadamard product weights of the input terms


def _register_linearizer(name: str):
    def decorator(linearizer: callable):
        def handler(*exprs: Expr) -> Expr:
            bounds = linearizer(*exprs)
            assert all(w.shape == e.shape for w, e in zip(bounds.input_weights, exprs)), \
                "Input weights must match the shapes of the input expressions."
            # Apply slopes to input expressions
            result_expr = sum(w * e for w, e in zip(bounds.input_weights, exprs) if w is not 0) + bounds.bias
            # Introduce a fresh noise symbol for the approximation error
            new_eps = LpEpsilon(bounds.error_coeffs.input_shape)
            result_expr = result_expr + bounds.error_coeffs(new_eps)
            return result_expr
        interpret.dispatcher[name] = handler
        return linearizer
    return decorator


# =====================================================================
# Affine operator registrations
# =====================================================================

def _linear(x, w, b=None):
    """aten.linear.default: weight and bias arrive as ConstVal from get_attr."""
    assert isinstance(w, expr.ConstVal), "Expected weight to arrive as ConstVal"
    assert b is None or isinstance(b, expr.ConstVal), "Expected bias to arrive as ConstVal or None"
    return x @ w.value.T + (b.value if b is not None else 0)

interpret.dispatcher.update({
    # ---- arithmetic ---------------------------------------------------
    "add":        lambda x, y: x + y,
    "sub":        lambda x, y: x - y,
    "neg":        lambda x: -x,
    "mul":        lambda x, y: x * y if isinstance(y, torch.Tensor) else y * x,
    "div":        lambda x, y: x / y,
    "truediv":    lambda x, y: x / y,
    "floordiv":   lambda x, y: x / y,
    # ---- linear layers ------------------------------------------------
    # F.linear / aten.linear.default
    "linear":     _linear,
    # ATen lowered form: aten.t.default + aten.addmm.default
    "t":          lambda x: x.transpose(0, 1),
    "addmm":      lambda bias, inp, mat2: (
                      inp @ (mat2.value if isinstance(mat2, expr.ConstVal) else mat2)
                      + (bias.value if isinstance(bias, expr.ConstVal) else bias)
                  ),
    # ---- shape ops ----------------------------------------------------
    "reshape":    lambda x, *shape: x.reshape(*shape),
    "view":       lambda x, *shape: x.reshape(*shape),
    "flatten":    lambda x, start_dim=0, end_dim=-1: x.flatten(start_dim, end_dim),
    "permute":    lambda x, *dims: x.permute(*dims),
    "transpose":  lambda x, dim0, dim1: x.transpose(dim0, dim1),
    "unsqueeze":  lambda x, dim: x.unsqueeze(dim),
    "squeeze":    lambda x, dim=None: x.squeeze(dim),
    "contiguous": lambda x: x,
    # ---- tuple/pair helpers -------------------------------------------
    # getitem: integer index used for tuple-unpacking in exported graphs
    "getitem":    lambda x, idx: x[idx],
})


# =====================================================================
# Import activation modules — each calls _register_linearizer
# =====================================================================

from . import relu as _relu            # registers "relu"
from . import exp as _exp              # registers "exp"
from . import reciprocal as _reciprocal  # registers "reciprocal"
from . import tanh as _tanh            # registers "tanh"

# call_module handlers (mod is the nn.Module instance)
interpret.dispatcher["ReLU"]  = lambda _, x: interpret.dispatcher["relu"](x)
interpret.dispatcher["Tanh"]  = lambda _, x: interpret.dispatcher["tanh"](x)

# Bilinear matmul handler (supports Expr @ Expr)
from .bilinear import matmul_handler, bilinear_matmul, bilinear_elementwise  # noqa: F401
interpret.dispatcher["matmul"] = matmul_handler
interpret.dispatcher["bmm"]    = matmul_handler
interpret.dispatcher["mm"]     = matmul_handler

# Softmax: both call_module (nn.Softmax) and ATen lowered (_softmax.default)
from .softmax import softmax_handler
interpret.dispatcher["softmax"]  = softmax_handler
interpret.dispatcher["_softmax"] = lambda x, dim, _half_to_float=False: softmax_handler(x, dim=dim)

from .relu import relu_linearizer
from .exp import exp_linearizer
from .reciprocal import reciprocal_linearizer
from .tanh import tanh_linearizer

__all__ = [
    "interpret", "ZonoBounds",
    "relu_linearizer", "exp_linearizer", "reciprocal_linearizer", "tanh_linearizer",
    "bilinear_matmul", "bilinear_elementwise", "matmul_handler", "softmax_handler",
]
