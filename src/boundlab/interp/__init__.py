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
    indices = _as_const(indices)

    # Handle DiffExpr3
    try:
        from boundlab.diff.expr import DiffExpr3
        if isinstance(data, DiffExpr3):
            return DiffExpr3(
                _onnx_gather(data.x, indices, axis),
                _onnx_gather(data.y, indices, axis),
                _onnx_gather(data.diff, indices, axis),
            )
    except ImportError:
        pass

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

    # DiffExpr support: evaluate component-wise
    try:
        from boundlab.diff.expr import DiffExpr2, DiffExpr3
        diff_positions = [i for i, v in enumerate(inputs) if isinstance(v, (DiffExpr3, DiffExpr2))]
    except Exception:
        DiffExpr2 = DiffExpr3 = ()
        diff_positions = []

    if diff_positions:
        if len(diff_positions) == 1:
            di = diff_positions[0]
            diff = inputs[di]
            if isinstance(diff, DiffExpr2):
                diff = DiffExpr3(diff.x, diff.y, diff.x - diff.y)
            in_x = list(inputs); in_x[di] = diff.x
            in_y = list(inputs); in_y[di] = diff.y
            in_d = list(inputs); in_d[di] = diff.diff
            out_x = _onnx_einsum(*in_x, equation=equation)
            out_y = _onnx_einsum(*in_y, equation=equation)
            out_d = _onnx_einsum(*in_d, equation=equation)
            return DiffExpr3(out_x, out_y, out_d)
        elif len(diff_positions) == 2:
            # Bilinear case: two DiffExpr inputs.
            # Use identity: A.x ⊗ B.x − A.y ⊗ B.y = A.diff ⊗ B.x + A.y ⊗ (B.x − B.y)
            # Identify DiffExpr3 (has .diff) and DiffExpr2 positions.
            di3 = [i for i in diff_positions if isinstance(inputs[i], DiffExpr3)]
            di2 = [i for i in diff_positions if isinstance(inputs[i], DiffExpr2)]
            in_x = list(inputs)
            in_y = list(inputs)
            for i in diff_positions:
                in_x[i] = _as_const(inputs[i].x) if isinstance(inputs[i], DiffExpr2) else inputs[i].x
                in_y[i] = _as_const(inputs[i].y) if isinstance(inputs[i], DiffExpr2) else inputs[i].y
            out_x = _onnx_einsum(*in_x, equation=equation)
            out_y = _onnx_einsum(*in_y, equation=equation)
            if len(di3) == 1 and len(di2) == 1:
                # Bilinear identity: diff = A.diff ⊗ B.x + A.y ⊗ (B.x − B.y)
                a_idx, b_idx = di3[0], di2[0]
                a, b = inputs[a_idx], inputs[b_idx]
                bx = _as_const(b.x)
                by = _as_const(b.y)
                in_d1 = list(inputs)
                in_d1[a_idx] = a.diff
                in_d1[b_idx] = bx
                in_d2 = list(inputs)
                in_d2[a_idx] = a.y
                in_d2[b_idx] = bx - by
                out_d = _onnx_einsum(*in_d1, equation=equation) + _onnx_einsum(*in_d2, equation=equation)
            else:
                out_d = out_x - out_y
            return DiffExpr3(out_x, out_y, out_d)
        else:
            raise NotImplementedError(f"Einsum with {len(diff_positions)} DiffExpr inputs is not supported")

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
    """ONNX Conv (2D), restricted to ``kernel_size == stride``.

    Reshapes ``X`` into non-overlapping patches
    ``[N, C_in, H/kH, kH, W/kW, kW]`` and contracts against the weight
    tensor ``W`` of shape ``[C_out, C_in, kH, kW]`` via :func:`_onnx_einsum`,
    which fuses with surrounding linear ops.
    """
    if kernel_shape is None:
        kernel_shape = list(W.shape[2:])
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
    Y = _onnx_einsum(X_r, W, equation="nchHwW,ocHW->nohw")
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
    slices = [slice(None)] * ndim
    for a, s, e, st in zip(axes_list, starts_list, ends_list, steps_list):
        if e > data.shape[a]:
            e = data.shape[a]
        step = st if st != 1 else None
        slices[a] = slice(s, e, step)
    return data[tuple(slices)]


def _onnx_concat(*args, axis=0):
    """ONNX Concat: concatenate inputs along axis."""
    inputs = list(args)
    from boundlab.expr._core import Expr
    from boundlab.expr._affine import ConstVal

    # Check for DiffExpr types
    try:
        from boundlab.diff.expr import DiffExpr2, DiffExpr3
        if any(isinstance(x, DiffExpr3) for x in inputs):
            from boundlab.expr import Cat
            x_parts, y_parts, d_parts = [], [], []
            for inp in inputs:
                if isinstance(inp, DiffExpr3):
                    x_parts.append(inp.x)
                    y_parts.append(inp.y)
                    d_parts.append(inp.diff)
                elif isinstance(inp, DiffExpr2):
                    x_parts.append(inp.x if isinstance(inp.x, Expr) else ConstVal(inp.x))
                    y_parts.append(inp.y if isinstance(inp.y, Expr) else ConstVal(inp.y))
                    shape = inp.x.shape if hasattr(inp.x, 'shape') else inp.y.shape
                    d_parts.append(ConstVal(torch.zeros(shape)))
                else:
                    v = inp if isinstance(inp, Expr) else ConstVal(inp) if isinstance(inp, torch.Tensor) else inp
                    x_parts.append(v)
                    y_parts.append(v)
                    d_parts.append(ConstVal(torch.zeros(v.shape)))
            dim = int(axis)
            return DiffExpr3(Cat(*x_parts, dim=dim), Cat(*y_parts, dim=dim), Cat(*d_parts, dim=dim))
    except ImportError:
        pass

    if any(isinstance(x, Expr) for x in inputs):
        from boundlab.expr import Cat
        wrapped = [x if isinstance(x, Expr) else ConstVal(x) for x in inputs]
        return Cat(*wrapped, dim=int(axis))
    return torch.cat(inputs, dim=int(axis))


def _onnx_broadcast(X, Y):
    """Broadcast X and Y to compatible shapes (ONNX numpy-style rules)."""
    def _get_shape(v):
        if hasattr(v, 'shape'):
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

def _onnx_expand(data, shape):
    """ONNX Expand: broadcast *data* to *shape*."""
    target = _unwrap_shape(shape)
    if hasattr(data, 'expand'):
        return data.expand(*target)
    elif isinstance(data, torch.Tensor):
        return data.expand(target)
    else:
        raise TypeError(f"Cannot expand object of type {type(data)}")

def _as_const(x):
    """Extract a concrete tensor from a DiffExpr2/3 or ConstVal for shape/index constants."""
    from boundlab.expr._affine import ConstVal as CV
    if isinstance(x, CV):
        return x.value
    try:
        from boundlab.diff.expr import DiffExpr2, DiffExpr3
        if isinstance(x, DiffExpr2):
            c = x.get_const()
            return c[0] if c is not None else _as_const(x.x)
        if isinstance(x, DiffExpr3):
            return _as_const(x.x)
    except ImportError:
        pass
    return x


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
            self.fns[0]
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

class FnListChain:

    def __init__(self, *fli: Callable[..., E]):
        self.fn_list = fli
    
    def __call__(self, *args: E, **kwargs) -> E:
        result = self.fn_list[0](*args, **kwargs)
        for fn in self.fn_list[1:]:
            result = fn(result, **kwargs)
        return result

# =====================================================================
# Interpreter
# =====================================================================

class Interpreter(Generic[E]):
    def __init__(self, dispatcher: dict[str, Callable[..., E]]):
        """Initialize an interpreter with a dispatcher.

        The dispatcher maps ONNX operator names to handler functions.

        Keys are the ONNX ``op_type`` strings (e.g. ``"Gemm"``, ``"Relu"``,
        ``"Reshape"``).  Custom-domain ops (e.g. ``"DiffPair"`` from the
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

    def and_then(self, other: Callable[[E], E], with_op_name: bool = False) -> Interpreter:
        """Return a new interpreter that applies another function to the output of this one."""
        result = {}

        class WithOpNameWrapper:
            def __init__(self, fn, op_name):
                self.fn = fn
                self.op_name = op_name

            def __call__(self, *args, **kwargs):
                return self.fn(*args, op_name=self.op_name, **kwargs)

        for k, v in self.dispatcher.items():
            if with_op_name:
                result[k] = FnListChain(v, WithOpNameWrapper(other, k))
            else:
                result[k] = FnListChain(v, other)
        return Interpreter(result)

    def __call__(
        self, model: ir.Model | str | Path, verbose: bool = False
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
              dispatched as ``"DiffPair"``.

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
                def to_repr(x: Any) -> str:
                    if isinstance(x, torch.Tensor):
                        return f"Tensor{list(x.shape)}({x.abs().max().item():.4g})"
                    return repr(x)
                
                if verbose:
                    outputs = ", ".join("%" + node.name for node in node.outputs if node is not None)
                    inputs = ", ".join("%" + node.name for node in node.inputs if node is not None)
                    kwargs_str = ", ".join(f"{k}={to_repr(v)}" for k, v in kwargs.items())
                    if kwargs_str:
                        kwargs_str = ", " + kwargs_str
                    print(f"{outputs} = {node.op_type}({inputs}{kwargs_str})")

                handler = self.dispatcher[node.op_type]
                result = handler(*args, **kwargs)
                

                if verbose:
                    arg_str = ", ".join(to_repr(arg) for arg in args)
                    print(f"{to_repr(result)} <- {arg_str}")

                assert not isinstance(result, tuple), f"Handler for {node.op_type} returned a tuple, but only single outputs are supported. Got: {result}"

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
    # ---- arithmetic (with broadcast) --------------------------------------
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
    "Gather":   _onnx_gather,
    "Slice":    _onnx_slice,
    "Concat":   _onnx_concat,
    "Identity": lambda X: X,
    "Cast":     _onnx_cast,
    "Expand":   _onnx_expand,
    # ---- reductions ---------------------------------------------------
    "ReduceSum": _onnx_reduce_sum,
    "ReduceMean": _onnx_reduce_mean,
    # ---- constants ---------------------------------------------
    "Constant": _onnx_constant,
    "Reciprocal": lambda X: 1 / X,
})
