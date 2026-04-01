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

from typing import Any, Callable, Generic, TypeVar
import torch
import torch.fx

from boundlab.expr._core import Expr

__all__ = ["Interpreter"]

E = TypeVar("E", bound=Expr)


class Interpreter(Generic[E]):
    def __init__(self, dispatcher: dict[str, Callable]):
        """Initialize an interpreter with a dispatcher.

        The dispatcher maps neural network operators to functions for interpretation.

        Keys are:

        - For ``call_function`` nodes: the callable's ``__name__``, with any
          ``.default`` overload suffix stripped (e.g.
          ``"linear.default"`` -> ``"linear"``).
        - For ``call_module`` nodes: the submodule's class name (e.g. ``"ReLU"``).
        - For ``call_method`` nodes: the method name string (e.g. ``"reshape"``).
        """
        self.dispatcher = dispatcher

    def __call__(
        self, model: torch.export.ExportedProgram | torch.fx.GraphModule
    ) -> Callable[..., E | tuple[E, ...]]:
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

        def interpret(*exprs: E) -> E | tuple[E, ...]:
            env: dict[str, Any] = {}
            input_idx = 0

            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    env[node.name] = exprs[input_idx]
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
