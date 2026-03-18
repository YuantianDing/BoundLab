"""Abstract Interpretation Framework for Neural Network Verification"""

from __future__ import annotations

from typing import Any, Callable
from torch import nn
import torch

from boundlab.expr._core import Expr

__all__ = ["Interpreter"]

class Interpreter:
    def __init__(self, dispatcher: dict[str, Callable], handle_affine: bool = True):
        """Initialize an interpreter with a dispatcher and an estimator.

        The dispatcher maps neural network operators to functions for interpretation.
        Keys are:
        - For call_function nodes: the callable's ``__name__`` (e.g. ``"add"``).
        - For call_module nodes: the submodule's class name (e.g. ``"Linear"``).
        - For call_method nodes: the method name string (e.g. ``"relu"``).
        """
        self.dispatcher = dispatcher
        if handle_affine:
            self.dispatcher = _AFFINE_DISPATCHER | dispatcher

    def __call__(self, model: nn.Module | torch.export.ExportedProgram) -> Callable:
        def interpret(*exprs: Expr) -> Expr | tuple[Expr, ...]:
            """Interpret the given model on the provided input expressions."""
            if isinstance(model, nn.Module):
                traced = torch.fx.symbolic_trace(model)
            else:
                traced = model

            env: dict[str, Any] = {}
            input_idx = 0

            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    env[node.name] = exprs[input_idx]
                    input_idx += 1

                elif node.op == "get_attr":
                    from boundlab.expr._base import ConstVal
                    obj = traced
                    for part in node.target.split("."):
                        obj = getattr(obj, part)
                    env[node.name] = ConstVal(obj.detach())

                elif node.op == "call_function":
                    key = node.target.__name__
                    handler = self.dispatcher[key]
                    args = [env[a.name] if isinstance(a, torch.fx.Node) else a for a in node.args]
                    kwargs = {k: (env[v.name] if isinstance(v, torch.fx.Node) else v) for k, v in node.kwargs.items()}
                    env[node.name] = handler(*args, **kwargs)

                elif node.op == "call_module":
                    submod = traced.get_submodule(node.target)
                    key = type(submod).__name__
                    handler = self.dispatcher[key]
                    args = [env[a.name] if isinstance(a, torch.fx.Node) else a for a in node.args]
                    kwargs = {k: (env[v.name] if isinstance(v, torch.fx.Node) else v) for k, v in node.kwargs.items()}
                    env[node.name] = handler(submod, *args, **kwargs)

                elif node.op == "call_method":
                    key = node.target
                    handler = self.dispatcher[key]
                    args = [env[a.name] if isinstance(a, torch.fx.Node) else a for a in node.args]
                    kwargs = {k: (env[v.name] if isinstance(v, torch.fx.Node) else v) for k, v in node.kwargs.items()}
                    env[node.name] = handler(*args, **kwargs)

                elif node.op == "output":
                    out = node.args[0]
                    if isinstance(out, torch.fx.Node):
                        return env[out.name]
                    return tuple(env[a.name] if isinstance(a, torch.fx.Node) else a for a in out)

        return interpret


_AFFINE_DISPATCHER: dict[str, Callable] = {
    # ---- arithmetic call_function ops ----------------------------------
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "neg": lambda x: -x,
    "mul": lambda x, y: x * y if isinstance(y, torch.Tensor) else y * x,
    # ---- call_module: linear layers ------------------------------------
    "Linear":       lambda mod, x: x @ mod.weight.T + mod.bias,
    # ---- call_function: F.linear (weight/bias arrive as ConstVal) ------
    "linear": lambda x, w, b=None: x @ w.value.T + (b.value if b is not None else 0),
    # ---- shape-manipulation call_method ops ----------------------------
    "reshape":    lambda x, *shape: x.reshape(*shape),
    "view":       lambda x, *shape: x.reshape(*shape),
    "flatten":    lambda x, start_dim=0, end_dim=-1: x.flatten(start_dim, end_dim),
    "permute":    lambda x, *dims: x.permute(*dims),
    "transpose":  lambda x, dim0, dim1: x.transpose(dim0, dim1),
    "unsqueeze":  lambda x, dim: x.unsqueeze(dim),
    "squeeze":    lambda x, dim=None: x.squeeze(dim),
    "contiguous": lambda x: x,
}
