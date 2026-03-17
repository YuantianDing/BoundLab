"""Abstract Interpretation Framework for Neural Network Verification"""


from typing import Any, Callable
from torch import nn
import torch

from boundlab.expr._core import Expr


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


def _conv2d_handler(mod, x):
    import torch.nn.functional as F
    from boundlab.linearop import LinearOp
    from boundlab.expr._base import ConstVal
    w = mod.weight.detach()
    stride, padding = mod.stride, mod.padding
    dilation, groups = mod.dilation, mod.groups
    lin_op = LinearOp(
        lambda inp: F.conv2d(inp, w, None, stride, padding, dilation, groups),
        x.shape, name="conv2d",
    )
    result = lin_op(x)
    if mod.bias is not None:
        b = mod.bias.detach()
        ndim = len(lin_op.output_shape)
        result = result + ConstVal(b.reshape(-1, *([1] * (ndim - 1))).expand(lin_op.output_shape))
    return result


def _batchnorm_handler(mod, x):
    from boundlab.expr._base import ConstVal
    scale = (mod.weight / (mod.running_var + mod.eps).sqrt()).detach()
    shift = (mod.bias - mod.running_mean * scale).detach()
    ndim = len(x.shape)
    if ndim > 1:
        scale = scale.reshape(-1, *([1] * (ndim - 1)))
        shift = shift.reshape(-1, *([1] * (ndim - 1)))
    return scale * x + ConstVal(shift)


def _avgpool2d_handler(mod, x):
    import torch.nn.functional as F
    from boundlab.linearop import LinearOp
    ks, stride = mod.kernel_size, mod.stride
    padding, ceil_mode = mod.padding, mod.ceil_mode
    count_include_pad = mod.count_include_pad
    lin_op = LinearOp(
        lambda inp: F.avg_pool2d(inp, ks, stride, padding, ceil_mode, count_include_pad),
        x.shape, name="avg_pool2d",
    )
    return lin_op(x)


_AFFINE_DISPATCHER: dict[str, Callable] = {
    # ---- arithmetic call_function ops ----------------------------------
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "neg": lambda x: -x,
    "mul": lambda x, y: x * y if isinstance(y, torch.Tensor) else y * x,
    # ---- call_module: linear layers ------------------------------------
    "Linear":       lambda mod, x: x @ mod.weight.T + mod.bias,
    "Conv2d":       _conv2d_handler,
    "BatchNorm1d":  _batchnorm_handler,
    "BatchNorm2d":  _batchnorm_handler,
    "AvgPool2d":    _avgpool2d_handler,
    # ---- call_function: F.linear (weight/bias arrive as ConstVal) ------
    "linear": lambda x, w, b=None: x @ w.value.T + (b.value if b is not None else 0),
    # ---- activations: placeholders overridden by zono.interpret --------
    "relu":      lambda x: x.ub().clamp(min=0),
    "sigmoid":   lambda x: torch.sigmoid(x.ub()),
    "tanh":      lambda x: torch.tanh(x.ub()),
    "ReLU":      lambda _, x: x.ub().clamp(min=0),
    "Sigmoid":   lambda _, x: torch.sigmoid(x.ub()),
    "Tanh":      lambda _, x: torch.tanh(x.ub()),
    "LeakyReLU": lambda mod, x: torch.nn.functional.leaky_relu(x.ub(), mod.negative_slope),
    "Hardtanh":  lambda mod, x: torch.clamp(x.ub(), mod.min_val, mod.max_val),
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
