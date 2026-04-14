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
from .onnx import onnx_export

__all__ = ["Interpreter", "ONNX_BASE_INTERPRETER", "onnx_export"]

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


def _onnx_reshape(data, shape, allowzero=0):
    del allowzero
    return data.reshape(_unwrap_shape(shape))


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

def _onnx_constant(value=None, value_float=None, value_int=None, value_string=None, value_floats=None, value_ints=None, value_strings=None, **_):
    """ONNX Constant node: wrap the tensor attribute as a torch.Tensor."""

    if value is not None:
        return torch.Tensor(value) if value is not None else None
    elif value_float is not None:
        return torch.tensor(value_float)
    elif value_int is not None:
        return torch.tensor(value_int)
    elif value_string is not None:
        return value_string
    elif value_floats is not None:
        return torch.tensor(value_floats)
    elif value_ints is not None:
        return torch.tensor(value_ints)
    elif value_strings is not None:
        return value_strings
    else:
        raise ValueError(f"ONNX Constant: no value provided among {locals()}")


_ONNX_DTYPE_MAP = {
    1: torch.float32,
    2: torch.uint8,
    3: torch.int8,
    5: torch.int16,
    6: torch.int32,
    7: torch.int64,
    9: torch.bool,
    10: torch.float16,
    11: torch.float64,
    16: torch.bfloat16,
}


def _onnx_cast(input, to):
    """ONNX Cast: convert tensor dtype."""
    dtype = _ONNX_DTYPE_MAP.get(to)
    if dtype is None:
        raise ValueError(f"Unsupported ONNX dtype id: {to}")
    return input.to(dtype)


def _normalize_reduce_axes(axes):
    if axes is None:
        return None
    return tuple(int(a) for a in _unwrap_shape(axes))


def _onnx_reduce_sum(data, axes=None, keepdims=1, noop_with_empty_axes=0):
    reduce_axes = _normalize_reduce_axes(axes)
    if reduce_axes == () and int(noop_with_empty_axes) == 1:
        return data
    return data.sum(dim=reduce_axes, keepdim=bool(keepdims))


def _onnx_reduce_mean(data, axes=None, keepdims=1, noop_with_empty_axes=0):
    reduce_axes = _normalize_reduce_axes(axes)
    if reduce_axes == () and int(noop_with_empty_axes) == 1:
        return data
    return data.mean(dim=reduce_axes, keepdim=bool(keepdims))


def _onnx_gather(data, indices, axis=0):
    axis = int(axis)
    if isinstance(data, Expr):
        rank = len(data.shape)
        axis = axis + rank if axis < 0 else axis
        idx = indices.long()
        if idx.numel() == 1:
            slices = [slice(None)] * rank
            slices[axis] = int(idx.item())
            return data[tuple(slices)]
        if idx.dim() == 1:
            from boundlab.expr import Cat
            parts = []
            for i in idx.tolist():
                slices = [slice(None)] * rank
                slices[axis] = int(i)
                parts.append(data[tuple(slices)].unsqueeze(axis))
            return Cat(*parts, dim=axis)
        raise NotImplementedError(
            f"ONNX Gather for Expr currently supports scalar/1D indices, got shape {tuple(idx.shape)}"
        )

    if indices.dim() == 0:
        index = indices.reshape(1).long()
    else:
        index = indices.long().reshape(-1)
    gathered = torch.index_select(data, axis, index)
    out_shape = list(data.shape[:axis]) + list(indices.shape) + list(data.shape[axis + 1 :])
    return gathered.reshape(out_shape)


def _onnx_concat(*inputs, axis):
    """ONNX Concat: concatenate inputs along *axis*.

    Dispatches to :class:`boundlab.expr.Cat` when any input is an
    :class:`Expr` (wrapping plain tensors as :class:`ConstVal`),
    otherwise uses :func:`torch.cat`.
    """
    axis = int(axis)
    if any(isinstance(x, Expr) for x in inputs):
        from boundlab.expr import Cat, ConstVal
        parts = [x if isinstance(x, Expr) else ConstVal(x) for x in inputs]
        return Cat(*parts, dim=axis)
    return torch.cat(list(inputs), dim=axis)


def _onnx_einsum(*inputs, equation):
    """ONNX Einsum: dispatch to torch.einsum for constants, or build an
    :class:`EinsumOp` when exactly one operand is an :class:`Expr`.

    Constant operands are pre-contracted into a single tensor whose axes span
    ``union(x_labels, out_labels)``, then wrapped by :class:`EinsumOp` so it
    fuses with surrounding linear ops.
    """
    equation = equation.replace(" ", "")
    lhs, rhs = equation.split("->")
    in_labels = lhs.split(",")
    assert len(in_labels) == len(inputs), f"Einsum equation mismatch: {equation}"

    expr_positions = [i for i, v in enumerate(inputs) if isinstance(v, Expr)]
    if not expr_positions:
        return torch.einsum(equation, *inputs)
    assert len(expr_positions) == 1, "Einsum with multiple Expr inputs is not supported"
    ei = expr_positions[0]
    x = inputs[ei]
    x_labels = in_labels[ei]
    const_labels = [in_labels[i] for i in range(len(inputs)) if i != ei]
    const_tensors = [inputs[i] for i in range(len(inputs)) if i != ei]
    out_labels = rhs

    t_labels = list(dict.fromkeys(x_labels + out_labels))
    sizes = {l: s for l, s in zip(x_labels, x.shape)}
    for t, lbl in zip(const_tensors, const_labels):
        for l, s in zip(lbl, t.shape):
            sizes.setdefault(l, s)

    const_label_set = set("".join(const_labels))
    contract_target = "".join(l for l in t_labels if l in const_label_set)
    if const_tensors:
        contract_eq = ",".join(const_labels) + "->" + contract_target
        tensor = torch.einsum(contract_eq, *const_tensors)
    else:
        tensor = torch.ones(())

    current = list(contract_target)
    for i, l in enumerate(t_labels):
        if l not in current:
            tensor = tensor.unsqueeze(i)
            current.insert(i, l)
    tensor = tensor.expand([sizes[l] for l in t_labels]).contiguous()

    input_dims = [t_labels.index(l) for l in x_labels]
    output_dims = [t_labels.index(l) for l in out_labels]

    from boundlab.linearop._einsum import EinsumOp
    from boundlab.expr._affine import AffineSum
    op = EinsumOp(tensor, input_dims, output_dims)
    return AffineSum((op, x))


def _onnx_conv(X, W, B=None, *, kernel_shape=None, strides=None, pads=None, dilations=None, group=1, auto_pad="NOTSET", **_):
    if kernel_shape is None:
        kernel_shape = list(W.shape[2:])
    """ONNX Conv (2D), restricted to ``kernel_size == stride``.

    Reshapes ``X`` into non-overlapping patches
    ``[N, C_in, H/kH, kH, W/kW, kW]`` and contracts against the weight
    tensor ``W`` of shape ``[C_out, C_in, kH, kW]`` via :func:`_onnx_einsum`,
    which fuses with surrounding linear ops.
    """
    assert len(kernel_shape) == 2, f"only 2D Conv is supported, got kernel_shape={kernel_shape}"
    kH, kW = int(kernel_shape[0]), int(kernel_shape[1])
    strides = [kH, kW] if strides is None else [int(s) for s in strides]
    assert strides == [kH, kW], f"Conv requires kernel_size == stride, got kernel={kernel_shape}, stride={strides}"
    assert int(group) == 1, "grouped Conv is not supported"
    assert auto_pad in ("NOTSET", "VALID"), f"Conv auto_pad={auto_pad} is not supported"
    if pads is not None:
        assert all(int(p) == 0 for p in pads), f"Conv with padding is not supported, got pads={pads}"
    if dilations is not None:
        assert all(int(d) == 1 for d in dilations), f"Conv with dilation is not supported, got dilations={dilations}"

    N, C_in, H, Win = X.shape
    C_out, C_in_w, kH_w, kW_w = W.shape
    assert C_in == C_in_w and kH == kH_w and kW == kW_w, \
        f"Conv weight shape {tuple(W.shape)} incompatible with input {tuple(X.shape)}"
    assert H % kH == 0 and Win % kW == 0, \
        f"input spatial dims ({H},{Win}) not divisible by kernel ({kH},{kW})"
    Hp, Wp = H // kH, Win // kW

    X_r = X.reshape(N, C_in, Hp, kH, Wp, kW)
    Y = _onnx_einsum(X_r, W, equation="nchHwW,ochW->nohw")
    if B is not None:
        Y = Y + B.reshape(1, C_out, 1, 1)
    return Y


def _onnx_slice(data, starts, ends, axes=None, steps=None, **_):
    """ONNX Slice: extract a slice from *data*."""
    starts_list = _unwrap_shape(_as_const(starts))
    ends_list = _unwrap_shape(_as_const(ends))
    ndim = len(data.shape)

    if axes is not None:
        axes_list = [a % ndim for a in _unwrap_shape(_as_const(axes))]
    else:
        axes_list = list(range(len(starts_list)))

    if steps is not None:
        steps_list = _unwrap_shape(_as_const(steps))
    else:
        steps_list = [1] * len(starts_list)

    # Build index tuple
    slices = [slice(None)] * ndim
    for a, s, e, st in zip(axes_list, starts_list, ends_list, steps_list):
        # ONNX uses INT_MAX-like sentinel for "to the end"
        if e > data.shape[a]:
            e = data.shape[a]
        step = st if st != 1 else None
        slices[a] = slice(s, e, step)

    return data[tuple(slices)]


# =====================================================================
# FnList — multi-handler dispatch helper
# =====================================================================

class FnList(Generic[E]):
    """Helper class for merging multiple handlers for the same operator."""
    def __init__(self, fns):
        if isinstance(fns, FnList):
            self.fns = copy.copy(fns.fns)
        elif isinstance(fns, list):
            self.fns = copy.copy(fns)
        else:
            self.fns = [fns]

    def __call__(self, *args: E, **kwargs) -> E:
        if len(self.fns) == 1:
            return self.fns[0](*args, **kwargs)
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

    def __getitem__(self, key) -> FnList[E]:
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
        result = {}

        for k, v in self.dispatcher.items():
            def chained_fn(*args, **kwargs):
                return other(v(*args, **kwargs))
            result[k] = chained_fn
        return Interpreter(result)

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
                torch.from_numpy(init.const_value.numpy().copy())
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
                assert e is not None, name
                env[name] = self.dispatcher["Input"](e)

            for node in model.graph:
                args = []
                for inp in node.inputs:
                    if inp is None:
                        args.append(None)
                        continue

                    inp_name = inp.name
                    if inp_name in env:
                        assert env[inp_name] is not None, inp_name
                        args.append(env[inp_name])
                    elif inp_name in initializers:
                        assert initializers[inp_name] is not None, inp_name
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

def _onnx_broadcast(X, Y):
    """Broadcast X and Y to compatible shapes (ONNX numpy-style rules)."""
    from boundlab.expr._core import Expr
    def _get_shape(v):
        if isinstance(v, (Expr, torch.Tensor)):
            return v.shape
        if hasattr(v, 'shape'):  # DiffExpr2, DiffExpr3
            return v.shape
        return ()
    x_shape = _get_shape(X)
    y_shape = _get_shape(Y)
    if x_shape == y_shape:
        return X, Y
    target = torch.broadcast_shapes(x_shape, y_shape)
    if hasattr(X, 'expand') and x_shape != target:
        X = X.expand(*target)
    elif isinstance(X, torch.Tensor) and X.shape != target:
        X = X.expand(target)
    if hasattr(Y, 'expand') and y_shape != target:
        Y = Y.expand(*target)
    elif isinstance(Y, torch.Tensor) and Y.shape != target:
        Y = Y.expand(target)
    return X, Y


def _as_const(x):
    """Extract a concrete tensor from a DiffExpr2/3 or pass through.

    Shape/axes/indices initializers get paired by diff_net but are
    identical in both branches — just take the first.
    """
    try:
        from boundlab.diff.expr import DiffExpr2, DiffExpr3
        if isinstance(x, DiffExpr2):
            c = x.get_const()
            return c[0] if c is not None else x.x
        if isinstance(x, DiffExpr3):
            return x.x
    except ImportError:
        pass
    return x


ONNX_BASE_INTERPRETER = Interpreter({
    "Input": lambda x, **_: x,
    "Initializer": lambda x, **_: x,
    # ---- arithmetic ---------------------------------------------------
    "Add":      lambda X, Y: (lambda a, b: a + b)(*_onnx_broadcast(X, Y)),
    "Sub":      lambda X, Y: (lambda a, b: a - b)(*_onnx_broadcast(X, Y)),
    "Neg":      lambda X: -X,
    "Mul":      lambda X, Y: (lambda a, b: a * b)(*_onnx_broadcast(X, Y)),
    "Div":      lambda X, Y: (lambda a, b: a / b)(*_onnx_broadcast(X, Y)),
    # ---- linear layers ------------------------------------------------
    "Gemm":     _onnx_gemm,
    "MatMul":   lambda A, B: A @ B,
    "Einsum":   _onnx_einsum,
    "Conv":     _onnx_conv,
    # ---- shape ops ----------------------------------------------------
    "Reshape":  lambda data, shape, **_: data.reshape(_unwrap_shape(_as_const(shape))),
    "Flatten":  _onnx_flatten,
    "Transpose": lambda data, perm=None: (data.permute(*perm) if perm is not None else data.T),
    "Unsqueeze": lambda data, axes: _onnx_unsqueeze(data, _as_const(axes)),
    "Squeeze":  lambda data, axes=None: _onnx_squeeze(data, _as_const(axes) if axes is not None else None),
    "Identity": lambda X: X,
    "Cast":     _onnx_cast,
    # ---- reductions ---------------------------------------------------
    "ReduceMean": _onnx_reduce_mean,
    "ReduceSum": _onnx_reduce_sum,
    # ---- indexing -----------------------------------------------------
    "Gather": _onnx_gather,
    "Slice": _onnx_slice,
    "Concat": _onnx_concat,
    # ---- constants ---------------------------------------------
    "Constant": _onnx_constant,
})
