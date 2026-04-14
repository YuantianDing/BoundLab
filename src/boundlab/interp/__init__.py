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


def _onnx_conv(X, W, B=None, auto_pad='NOTSET', dilations=None, group=1,
               kernel_shape=None, pads=None, strides=None, **_):
    """ONNX Conv: convolution. Supports Expr and DiffExpr inputs."""
    import torch.nn.functional as F

    # Handle DiffExpr3 input with DiffExpr2 weights
    try:
        from boundlab.diff.expr import DiffExpr2, DiffExpr3
        if isinstance(X, DiffExpr3):
            W_x = _as_const(W.x) if isinstance(W, DiffExpr2) else (_as_const(W) if not isinstance(W, torch.Tensor) else W)
            W_y = _as_const(W.y) if isinstance(W, DiffExpr2) else W_x
            if B is not None:
                B_x = _as_const(B.x) if isinstance(B, DiffExpr2) else (_as_const(B) if not isinstance(B, torch.Tensor) else B)
                B_y = _as_const(B.y) if isinstance(B, DiffExpr2) else B_x
            else:
                B_x = B_y = None
            kwargs = dict(auto_pad=auto_pad, dilations=dilations, group=group,
                          kernel_shape=kernel_shape, pads=pads, strides=strides)
            x_out = _onnx_conv(X.x, W_x, B_x, **kwargs)
            y_out = _onnx_conv(X.y, W_y, B_y, **kwargs)
            # diff = x - y, conv is linear: conv(diff, W_x) + conv(y, W_x - W_y)
            diff_out = _onnx_conv(X.diff, W_x, None, **kwargs)
            if not torch.equal(W_x, W_y):
                dW = W_x - W_y
                dB = (B_x - B_y) if (B_x is not None and B_y is not None) else None
                diff_out = diff_out + _onnx_conv(X.y, dW, dB, **kwargs)
            return DiffExpr3(x_out, y_out, diff_out)
    except ImportError:
        pass

    # Conv weight/bias may be DiffExpr2 from diff_net — extract concrete tensors
    W_concrete = _as_const(W) if not isinstance(W, torch.Tensor) else W
    B_concrete = _as_const(B) if B is not None and not isinstance(B, torch.Tensor) else B

    dilations = dilations or [1] * (W_concrete.dim() - 2)
    strides = strides or [1] * (W_concrete.dim() - 2)
    if pads is None:
        pads = [0] * (2 * (W_concrete.dim() - 2))
    ndim = W_concrete.dim() - 2
    padding = tuple(pads[i] for i in range(ndim))  # just take begin pads (symmetric)

    if isinstance(X, Expr):
        # Conv is linear: implement via ConvOp LinearOp
        # Input may be (C_in, H, W) or (1, C_in, H, W) with batch dim
        x_shape = X.shape
        has_batch = (len(x_shape) == ndim + 2)
        if has_batch:
            assert x_shape[0] == 1, f"Conv Expr only supports batch=1, got {x_shape[0]}"
            spatial_shape = x_shape[1:]  # (C_in, H, W)
        else:
            spatial_shape = x_shape  # (C_in, H, W)

        assert len(spatial_shape) == ndim + 1, f"Only 2D conv supported for Expr, got {ndim}D"
        C_in = spatial_shape[0]
        H_in, W_in = spatial_shape[1], spatial_shape[2]
        C_out = W.shape[0]
        kH, kW = W.shape[2], W.shape[3]
        sH, sW = strides[0], strides[1]
        pH, pW = padding[0], padding[1]
        dH, dW = dilations[0], dilations[1]

        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        # Build the im2col → matmul as a concrete operation on a meta tensor,
        # then construct the EinsumOp from the kernel weights
        # unfold: (1, C_in, H, W) → (1, C_in*kH*kW, L) where L = H_out*W_out
        # matmul: W_reshaped (C_out, C_in*kH*kW) @ unfolded (C_in*kH*kW, L) → (C_out, L)
        # reshape → (C_out, H_out, W_out)

        from boundlab.linearop._base import LinearOp, LinearOpFlags

        class ConvOp(LinearOp):
            def __init__(self, weight, bias, stride, pad, dilation, groups, in_shape, out_shape):
                self.weight = weight
                self.conv_bias = bias
                self.stride = stride
                self.pad = pad
                self.dilation = dilation
                self.groups = groups
                super().__init__(in_shape, out_shape, flags=LinearOpFlags.NONE)

            def forward(self, x):
                y = F.conv2d(x.unsqueeze(0), self.weight, None,
                             stride=self.stride, padding=self.pad,
                             dilation=self.dilation, groups=self.groups).squeeze(0)
                return y

            def backward(self, grad):
                return F.conv_transpose2d(
                    grad.unsqueeze(0), self.weight, None,
                    stride=self.stride, padding=self.pad,
                    dilation=self.dilation, groups=self.groups
                ).squeeze(0)

            def vforward(self, x):
                # x: (C_in, H, W, *batch)
                extra = x.shape[len(self.input_shape):]
                flat = x.reshape(*self.input_shape, -1)
                # Process each batch element
                results = []
                for i in range(flat.shape[-1]):
                    results.append(self.forward(flat[..., i]))
                out = torch.stack(results, dim=-1)
                return out.reshape(*self.output_shape, *extra)

            def vbackward(self, grad):
                # grad: (*batch, C_out, H_out, W_out)
                extra = grad.shape[:-len(self.output_shape)]
                flat = grad.reshape(-1, *self.output_shape)
                results = []
                for i in range(flat.shape[0]):
                    results.append(self.backward(flat[i]))
                out = torch.stack(results, dim=0)
                return out.reshape(*extra, *self.input_shape)

            def __str__(self):
                return f"conv2d({list(self.weight.shape)})"

        out_shape = torch.Size([C_out, H_out, W_out])
        if has_batch:
            # Strip batch dim, apply ConvOp on (C_in, H, W), add batch back
            from boundlab.linearop import SqueezeOp, UnsqueezeOp
            stripped = X._apply_op(SqueezeOp(x_shape, 0))
            op = ConvOp(W, B, strides, padding, dilations, int(group), 
                        torch.Size(spatial_shape), out_shape)
            result = stripped._apply_op(op)
            if B is not None:
                result = result + B.reshape(C_out, 1, 1).expand(out_shape)
            result = result._apply_op(UnsqueezeOp(out_shape, 0))
        else:
            op = ConvOp(W, B, strides, padding, dilations, int(group), x_shape, out_shape)
            result = X._apply_op(op)
            if B is not None:
                result = result + B.reshape(C_out, 1, 1).expand(out_shape)
        return result
    else:
        # Concrete tensor path
        y = F.conv2d(X.unsqueeze(0), W, B, stride=strides, padding=padding,
                     dilation=dilations, groups=int(group)).squeeze(0)
        return y


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
                assert e is not None, name
                env[name] = self.dispatcher["Input"](e, name=name)

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
    "Cast":     lambda input, to=None, **_: input,
    # ---- reductions ---------------------------------------------------
    "ReduceSum": _onnx_reduce_sum,
    "ReduceMean": _onnx_reduce_mean,
    # ---- constants ---------------------------------------------
    "Constant": _onnx_constant,
})
