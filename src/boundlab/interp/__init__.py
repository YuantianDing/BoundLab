"""Abstract Interpretation Framework for Neural Network Verification.

Examples
--------
Export a model then interpret it:

>>> import torch
>>> from torch import nn
>>> import boundlab.expr as expr
>>> from boundlab.interp import Interpreter
>>> itp = Interpreter({"relu": lambda x: x})
>>> gm = torch.export.export(nn.ReLU(), (torch.zeros(1),))
>>> op = itp(gm)
>>> x = expr.ConstVal(torch.tensor([0.0])) + expr.LpEpsilon([1])
>>> y = op(x)
>>> y.shape
torch.Size([1])
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Generic, TypeVar
import torch
import torch.fx
from typing import get_type_hints
import beartype
import beartype.roar

from boundlab.expr._core import Expr

__all__ = ["Interpreter"]

E = TypeVar("E", bound=Expr)


class FnList:
    """Helper class for merging multiple handlers for the same operator."""
    def __init__(self, fns):
        if isinstance(fns, FnList):
            self.fns = copy.copy(fns.fns)
        elif isinstance(fns, list):
            self.fns = copy.copy(fns)
        else:
            self.fns = [fns]

    def __call__(self, *args, **kwargs):
        errors = []
        for fn in self.fns[::-1]:
            try:
                result = beartype.beartype()(fn)(*args, **kwargs)
                if result is not NotImplemented:
                    return result
            except beartype.roar.BeartypeException as e:
                errors.append(e)
                continue
            except NotImplementedError as e:
                errors.append(e)
                continue
        raise TypeError(f"No matching handler found for arguments {args} {kwargs}. Errors: {errors}")
    
    def __add__(self, other: Callable[..., E] | FnList) -> FnList:
        if isinstance(other, FnList):
            return FnList(self.fns + other.fns)
        return FnList(self.fns + [other])
    
    def product(self, *other: FnList) -> FnList:
        zip_list = [self] + list(other)
        def zipped_fn(*args, **kwargs):
            results = (None,) * len(zip_list)
            for i in range(len(zip_list)):
                argsi = [args[i] for i in range(len(args))]
                kwargsi = {k: kwargs[k][i] for k in kwargs}
                results[i] = zip_list[i](*argsi, **kwargsi)
            return tuple(results)
        return FnList(zipped_fn)
        

class Interpreter(Generic[E]):
    def __init__(self, dispatcher: dict[str, Callable[..., E]]):
        """Initialize an interpreter with a dispatcher.

        The dispatcher maps neural network operators to functions for interpretation.

        Keys are:

        - For ``call_function`` nodes: the callable's ``__name__``, with any
          ``.default`` overload suffix stripped (e.g.
          ``"linear.default"`` -> ``"linear"``).
        - For ``call_module`` nodes: the submodule's class name (e.g. ``"ReLU"``).
        - For ``call_method`` nodes: the method name string (e.g. ``"reshape"``).
        """
        if isinstance(dispatcher, Interpreter):
            self.dispatcher = {k: FnList(v) for k, v in dispatcher.dispatcher.items()}
        else:
            self.dispatcher = {k: FnList(v) for k, v in dispatcher.items()}

    def __getitem__(self, key) -> FnList:
        return self.dispatcher[key]
    
    def __setitem__(self, key, value):
        if isinstance(value, FnList):
            for fn in value.fns:
                self.register(key, fn)
        else:
            self.register(key, value)

    def register(self, key: str, value: Callable[..., E]):
        """Register a handler for an operator."""
        assert callable(value), "Handler must be callable"
        if key in self.dispatcher:
            self.dispatcher[key].fns.append(value)
        else:
            self.dispatcher[key] = FnList(value)

    def __contains__(self, key) -> bool:
        return key in self.dispatcher
    
    def items(self):
        return self.dispatcher.items()
    
    def __or__(self, other: Interpreter | dict[str, Callable[..., E]]) -> Interpreter:
        result = Interpreter(self.dispatcher).deepcopy()
        result |= other
        return result
        
    def __ior__(self, other: Interpreter | dict[str, Callable[..., E]]):
        other = other if isinstance(other, Interpreter) else Interpreter(other)
        for k, v in other.items():
            for fn in v.fns:
                self.register(k, fn)
        return self
    
    def product(self, *other: Interpreter) -> Interpreter:
        """Return a new interpreter that produces tuples of results from this and other interpreters."""
        return Interpreter({k: v.product(*[o[k] for o in other]) for k, v in self.dispatcher.items()})

    def and_then(self, other: Callable[[E], E]) -> Interpreter:
        """Return a new interpreter that applies another function to the output of this one."""
        return Interpreter({k: lambda *args, **kwargs: other(v(*args, **kwargs)) for k, v in self.dispatcher.items()})

    def __call__(
        self, model: torch.export.ExportedProgram | torch.fx.GraphModule
    ) -> Callable[..., E]:
        """Build an expression-level interpreter for an exported model.

        Parameters
        ----------
        model:
            A :class:`torch.export.ExportedProgram` (e.g. from
            ``torch.export.export``) or an already-unwrapped
            :class:`torch.fx.GraphModule`.  Plain ``nn.Module`` is not accepted —
            export the model first so that ``call_module`` nodes for built-in layers
            are lowered to ``call_function`` nodes (e.g. ``linear.default``).

        Returns
        -------
        A callable ``interpret(*exprs)`` that maps input
        :class:`~boundlab.expr.Expr` objects to output expression(s).

        Examples
        --------
        >>> import torch
        >>> from torch import nn
        >>> import boundlab.expr as expr
        >>> import boundlab.zono as zono
        >>> model = nn.Linear(4, 3)
        >>> gm = torch.export.export(model, (torch.zeros(4),))
        >>> op = zono.interpret(gm)
        >>> x = expr.ConstVal(torch.zeros(4)) + expr.LpEpsilon([4])
        >>> y = op(x)
        >>> y.ub().shape
        torch.Size([3])
        """
        if isinstance(model, torch.export.ExportedProgram):
            traced = model.module()
        else:
            traced = model
        assert all(isinstance(v, FnList) for v in self.dispatcher.values()), "All handlers must be non-None."
        def interpret(*exprs: E) -> E | tuple[E, ...]:
            env: dict[str, Any] = {}
            input_idx = 0

            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    env[node.name] = self.dispatcher["placeholder"](exprs[input_idx], name=node.name)
                    input_idx += 1

                elif node.op == "get_attr":
                    from boundlab.expr._affine import ConstVal
                    obj = traced
                    for part in node.target.split("."):
                        obj = getattr(obj, part)
                    env[node.name] = ConstVal(obj.detach())

                elif node.op == "call_function":
                    raw = node.target.__name__
                    # Examples:
                    #   "aten.linear.default"        → "linear"
                    #   "boundlab.diff_pair.default" → "diff_pair"
                    #   "aten.transpose.int"         → "transpose"
                    #   "getitem"                    → "getitem"
                    parts = raw.split(".")
                    if len(parts) == 1:
                        key = parts[0]
                    elif parts[-1] == "default":
                        key = parts[-2]
                    else:
                        key = parts[0]
                    handler = self.dispatcher[key]
                    args = [env[a.name] if isinstance(a, torch.fx.Node) else a for a in node.args]
                    kwargs = {k: (env[v.name] if isinstance(v, torch.fx.Node) else v) for k, v in node.kwargs.items()}
                    env[node.name] = handler(*args, **kwargs)

                elif node.op == "call_module":
                    submod = traced.get_submodule(node.target)
                    key = type(submod).__name__
                    if key not in self.dispatcher:
                        if node.users:
                            raise KeyError(f"No handler for call_module '{key}'")
                        continue  # side-effect-only node (e.g. GuardsFn)
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
                    vals = tuple(env[a.name] if isinstance(a, torch.fx.Node) else a for a in out)
                    return vals[0] if len(vals) == 1 else vals

        return interpret

TENSOR_BASE_INTERPRETER = Interpreter({
    "placeholder": lambda x, name: x,
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
    "linear":     lambda x, w, b=None: x @ w.T + (b if b is not None else 0),
    # ATen lowered form: aten.t.default + aten.addmm.default
    "t":          lambda x: x.transpose(0, 1),
    "addmm":      lambda bias, inp, mat2: inp @ mat2 + bias,
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