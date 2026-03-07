"""Abstract Interpretation Framework for Neural Network Verification"""


from typing import Any, Callable
from torch import nn
import torch

from boundlab.expr._core import Expr

class Interpreter:
    def __init__(self, dispatcer: dict[str, Callable], handle_affine: bool = True):
        """Initialize an interpreter with a dispatcher and an estimator. 
        The dispatcher maps neural network operators to functions for interpretation.
        """
        self.dispatcer = dispatcer
        if handle_affine:
            self.dispatcer |= _AFFINE_DISPATCHER
    
    def __call__(self, model: nn.Module | torch.export.ExportedProgram) -> Callable:
        def interpret(*exprs: Expr) -> Expr:
            """Interpret the given model on the provided input expressions."""
            if isinstance(model, nn.Module):
                traced = torch.export.export(model, [e.center() for e in exprs])
            else:
                traced = model
            # TODO: Write the actual interpretation logic using the dispatcher to handle each operator in the traced graph.
            pass
        return interpret

_AFFINE_DISPATCHER = {
    "operator.add": lambda x, y: x + y,
    "linear": lambda x, weight, bias: weight @ x + bias,
    # TODO: Add as much as affine operator support as possible, including convolution, linear, etc.
}
