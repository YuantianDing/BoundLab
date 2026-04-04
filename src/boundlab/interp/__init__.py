"""Abstract Interpretation Framework for Neural Network Verification.

Examples
--------
Interpret an ONNX model:

>>> import torch
>>> import onnx_ir as ir
>>> import boundlab.expr as expr
>>> from boundlab.interp import Interpreter
>>> itp = Interpreter({"placeholder": lambda x, name: x, "Relu": lambda x: x})
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar
import onnx_ir as ir
import torch
import beartype
import beartype.roar

from boundlab.expr._core import Expr, ExprFlags

__all__ = ["Interpreter", "ONNX_BASE_INTERPRETER"]

E = TypeVar("E", bound=Expr)


# =====================================================================
# ONNX attribute / shape helpers (used by __call__ and ONNX_BASE_INTERPRETER)
# =====================================================================

def _onnx_attr_value(attr) -> Any:
    """Convert an ONNX IR attribute to a Python value."""
    t = attr.type
    if t == ir.AttributeType.FLOAT:
        return attr.as_float()
    elif t == ir.AttributeType.INT:
        return attr.as_int()
    elif t == ir.AttributeType.STRING:
        return attr.as_string()
    elif t == ir.AttributeType.TENSOR:
        return torch.from_numpy(attr.as_tensor().numpy().copy())
    elif t == ir.AttributeType.FLOATS:
        return list(attr.as_floats())
    elif t == ir.AttributeType.INTS:
        return list(attr.as_ints())
    elif t == ir.AttributeType.STRINGS:
        return list(attr.as_strings())
    else:
        return None


def _unwrap_shape(x) -> list[int]:
    """Extract a concrete shape/axes list from a tensor."""
    if isinstance(x, torch.Tensor):
        return x.long().tolist()
    return list(x)


def _onnx_gemm(A, B, C=None, alpha=1.0, beta=1.0, transA=0, transB=0):
    """ONNX Gemm: ``Y = alpha * (A' @ B') + beta * C``."""
    a = A.transpose(0, 1) if transA else A
    b = B.transpose(0, 1) if transB else B
    y = a @ b
    if alpha != 1.0:
        y = alpha * y
    if C is not None:
        y = y + (beta * C if beta != 1.0 else C)
    return y


def _onnx_flatten(X, axis=1):
    """ONNX Flatten: produce 2-D tensor ``[prod(dims[:axis]), prod(dims[axis:])]``."""
    first = math.prod(X.shape[:axis]) if axis > 0 else 1
    return X.reshape(first, -1)


def _onnx_unsqueeze(data, axes):
    """ONNX Unsqueeze: insert size-1 dims at the given axes positions."""
    axes_list = sorted(_unwrap_shape(axes))
    result = data
    for ax in axes_list:
        result = result.unsqueeze(ax)
    return result


def _onnx_squeeze(data, axes=None):
    """ONNX Squeeze: remove size-1 dims at *axes* (or all if omitted)."""
    if axes is None:
        return data.squeeze()
    axes_list = sorted(_unwrap_shape(axes), reverse=True)
    result = data
    for ax in axes_list:
        result = result.squeeze(ax)
    return result

def _onnx_constant(value=None, **_):
    """ONNX Constant node: wrap the tensor attribute as a torch.Tensor."""
    return torch.Tensor(value) if value is not None else None


# =====================================================================
# FnList — multi-handler dispatch helper
# =====================================================================

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


# =====================================================================
# Interpreter
# =====================================================================

class Interpreter(Generic[E]):
    def __init__(self, dispatcher: dict[str, Callable[..., E]]):
        """Initialize an interpreter with a dispatcher.

        The dispatcher maps ONNX operator names to handler functions.

        Keys are the ONNX ``op_type`` strings (e.g. ``"Gemm"``, ``"Relu"``,
        ``"Reshape"``).  Custom-domain ops (e.g. ``"diff_pair"`` from the
        ``boundlab`` domain) are also keyed by bare ``op_type``.
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
        self, model: ir.Model | str | Path
    ) -> Callable[..., E]:
        """Build an expression-level interpreter for an ONNX model.

        Parameters
        ----------
        model:
            An ``onnx_ir.Model`` or a ``str`` / :class:`pathlib.Path`
            pointing to an ``.onnx`` file.

            The ONNX graph is walked in topological order (ONNX guarantees
            this).  For each node:

            * Initializer inputs are wrapped as
              :class:`~torch.Tensor` and passed as positional
              arguments.
            * Optional/missing inputs (empty-string name) are passed as
              ``None``.
            * Node attributes are converted to Python scalars / lists and
              passed as keyword arguments.
            * The dispatcher is keyed on the bare ``op_type`` (domain is
              ignored); e.g. a custom ``boundlab::diff_pair`` node is
              dispatched as ``"diff_pair"``.

        Returns
        -------
        A callable ``interpret(*exprs)`` that maps input
        :class:`~boundlab.expr.Expr` objects to output expression(s).

        Examples
        --------
        >>> import torch, tempfile, os
        >>> from boundlab.interp import Interpreter, ONNX_BASE_INTERPRETER
        >>> from boundlab.zono import interpret
        >>> import boundlab.expr as expr
        """
        if isinstance(model, torch.onnx.ONNXProgram):
            model = model.model
        elif isinstance(model, torch.export.ExportedProgram):
            model = torch.onnx.export(model, dynamo=True).model

        assert isinstance(model, (ir.Model, str, Path)), "Model must be an onnx_ir.Model, ExportedProgram, ONNXProgram, or file path"

        if isinstance(model, (str, Path)):
            model = ir.load(str(model))

        initializers = {
            init.name: self.dispatcher["Initializer"](
                torch.from_numpy(init.const_value.numpy().copy()),
                name=init.name
            )
            for init in model.graph.initializers.values()
        }
        initializer_names = set(initializers.keys())
        input_names = [
            inp.name for inp in model.graph.inputs
            if inp.name not in initializer_names
        ]
        output_names = [out.name for out in model.graph.outputs]

        assert all(isinstance(v, FnList) for v in self.dispatcher.values()), \
            "All handlers must be non-None."

        def interpret(*exprs: E) -> E | tuple[E, ...]:
            env: dict[str, Any] = {}

            for name, e in zip(input_names, exprs):
                env[name] = self.dispatcher["Input"](e, name=name)

            for node in model.graph:
                args = []
                for inp in node.inputs:
                    if inp is None:
                        args.append(None)
                        continue

                    inp_name = inp.name
                    if inp_name in env:
                        args.append(env[inp_name])
                    elif inp_name in initializers:
                        args.append(initializers[inp_name])
                    else:
                        raise KeyError(
                            f"Input '{inp_name}' not found for node "
                            f"'{node.op_type}' ({node.name!r})"
                        )

                kwargs = {
                    name: _onnx_attr_value(attr)
                    for name, attr in node.attributes.items()
                }

                # Dispatch on op_type (ignore domain)
                handler = self.dispatcher[node.op_type]
                result = handler(*args, **kwargs)

                # Bind outputs
                if len(node.outputs) == 1:
                    out = node.outputs[0]
                    if out is not None and out.name:
                        env[out.name] = result
                else:
                    for i, out in enumerate(node.outputs):
                        if out is not None and out.name:
                            env[out.name] = result[i]

            outputs = [env[name] for name in output_names]
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        return interpret


# =====================================================================
# ONNX base interpreter
# =====================================================================

ONNX_BASE_INTERPRETER = Interpreter({
    "Input": lambda x, **_: x,
    "Initializer": lambda x, **_: x,
    # ---- arithmetic ---------------------------------------------------
    "Add":      lambda X, Y: X + Y,
    "Sub":      lambda X, Y: X - Y,
    "Neg":      lambda X: -X,
    "Mul":      lambda X, Y: X * Y,
    "Div":      lambda X, Y: X / Y,
    # ---- linear layers ------------------------------------------------
    "Gemm":     _onnx_gemm,
    "MatMul":   lambda A, B: A @ B,
    # ---- shape ops ----------------------------------------------------
    "Reshape":  lambda data, shape: data.reshape(_unwrap_shape(shape)),
    "Flatten":  _onnx_flatten,
    "Transpose": lambda data, perm=None: (data.permute(*perm) if perm is not None else data.T),
    "Unsqueeze": _onnx_unsqueeze,
    "Squeeze":  _onnx_squeeze,
    "Identity": lambda X: X,
    # ---- constants ---------------------------------------------
    "Constant": _onnx_constant,
})
