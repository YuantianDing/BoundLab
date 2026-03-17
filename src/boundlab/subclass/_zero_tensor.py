"""ZeroTensor: a tensor subclass that is always zero, using __torch_dispatch__."""

from __future__ import annotations

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

__all__ = ["ZeroTensor"]

_ZERO_TENSOR_PASSTHROUGH = {
    # Unary ops that preserve zero
    torch.ops.aten.neg.default,
    torch.ops.aten.abs.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.ceil.default,
    torch.ops.aten.floor.default,
    torch.ops.aten.round.default,
    torch.ops.aten.trunc.default,
    torch.ops.aten.sign.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.sinh.default,
    torch.ops.aten.tan.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.asin.default,
    torch.ops.aten.atan.default,
    torch.ops.aten.erf.default,
    torch.ops.aten.erfc.default,
    torch.ops.aten.clone.default,
}

# Ops where zero * anything = zero (first arg is zero tensor)
_ZERO_ABSORBING_LEFT = {
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten.mv.default,
    torch.ops.aten.dot.default,
}

# Ops where anything * zero = zero (second arg is zero tensor)
_ZERO_ABSORBING_RIGHT = {
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.matmul.default,
}

# View/reshape ops (out-of-place): zero in → zero out with new shape
_RESHAPE_OPS = {
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.unflatten.int,
    torch.ops.aten.narrow.default,
    torch.ops.aten.repeat.default,
}

# In-place reshape ops: need return_and_correct_aliasing
_INPLACE_RESHAPE_OPS = {
    torch.ops.aten.squeeze_.dim,
    torch.ops.aten.unsqueeze_.default if hasattr(torch.ops.aten, 'unsqueeze_') else None,
    torch.ops.aten.t_.default if hasattr(torch.ops.aten, 't_') else None,
}
_INPLACE_RESHAPE_OPS.discard(None)


def _infer_shape(func, args, kwargs):
    """Use meta tensors to infer the output shape of an operation."""
    def _to_meta(x):
        if isinstance(x, ZeroTensor):
            return torch.empty(x.shape, dtype=x.dtype, device="meta")
        if isinstance(x, torch.Tensor):
            return torch.empty(x.shape, dtype=x.dtype, device="meta")
        return x

    meta_args = torch.utils._pytree.tree_map(_to_meta, args)
    meta_kwargs = torch.utils._pytree.tree_map(_to_meta, kwargs)
    meta_out = func(*meta_args, **meta_kwargs)
    return meta_out


class ZeroTensor(torch.Tensor):
    """A tensor subclass that represents a zero tensor without storing data.

    ZeroTensor propagates through most linear operations, returning new
    ZeroTensors of the correct shape. For operations where zero-propagation
    isn't applicable, it materializes into a real zero tensor.

    Usage:
        z = ZeroTensor(torch.Size([3, 4]), dtype=torch.float32)
        result = z @ torch.randn(4, 5)  # returns ZeroTensor of shape [3, 5]
    """

    @staticmethod
    def __new__(cls, shape: torch.Size, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        return r

    def __init__(self, shape: torch.Size, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        pass

    def __repr__(self):
        return f"ZeroTensor(shape={list(self.shape)}, dtype={self.dtype})"

    def _is_zerotensor(self):
        """Compatibility with torch._efficientzerotensor checks."""
        return True

    def materialize(self) -> torch.Tensor:
        """Convert to a real zero tensor."""
        return torch.zeros(self.shape, dtype=self.dtype, device=self.device)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def _unwrap(x):
            if isinstance(x, ZeroTensor):
                return x.materialize()
            return x

        def _has_zero(*indices):
            """Check if any of the positional args at given indices is a ZeroTensor."""
            for i in indices:
                if i < len(args) and isinstance(args[i], ZeroTensor):
                    return True
            return False

        def _make_zero_like(meta_out):
            """Create ZeroTensor(s) matching the shape/dtype of meta output."""
            if isinstance(meta_out, torch.Tensor):
                dev = meta_out.device if meta_out.device.type != "meta" else "cpu"
                return ZeroTensor(meta_out.shape, dtype=meta_out.dtype, device=dev)
            if isinstance(meta_out, (tuple, list)):
                return type(meta_out)(_make_zero_like(o) for o in meta_out)
            return meta_out

        # --- out-of-place reshape / view ops ---
        if func in _RESHAPE_OPS and _has_zero(0):
            meta_out = _infer_shape(func, args, kwargs)
            return _make_zero_like(meta_out)

        # --- in-place reshape ops: must use return_and_correct_aliasing ---
        if func in _INPLACE_RESHAPE_OPS and _has_zero(0):
            meta_out = _infer_shape(func, args, kwargs)
            new_zero = _make_zero_like(meta_out)
            return return_and_correct_aliasing(func, args, kwargs, new_zero)

        # --- unary zero-preserving ops ---
        if func in _ZERO_TENSOR_PASSTHROUGH and _has_zero(0):
            meta_out = _infer_shape(func, args, kwargs)
            return _make_zero_like(meta_out)

        # --- zero * anything = zero ---
        if func in _ZERO_ABSORBING_LEFT and _has_zero(0):
            meta_out = _infer_shape(func, args, kwargs)
            return _make_zero_like(meta_out)

        # --- anything * zero = zero ---
        if func in _ZERO_ABSORBING_RIGHT and len(args) > 1 and isinstance(args[1], ZeroTensor):
            meta_out = _infer_shape(func, args, kwargs)
            return _make_zero_like(meta_out)

        # --- add/sub: zero + x = x, x + zero = x ---
        if func in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
            if _has_zero(0) and len(args) > 1 and not isinstance(args[1], ZeroTensor):
                other = args[1]
                if isinstance(other, torch.Tensor):
                    return other.clone()
                unwrapped_args = torch.utils._pytree.tree_map(_unwrap, args)
                unwrapped_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
                return func(*unwrapped_args, **unwrapped_kwargs)
            if len(args) > 1 and isinstance(args[1], ZeroTensor) and not isinstance(args[0], ZeroTensor):
                other = args[0]
                if isinstance(other, torch.Tensor):
                    return other.clone()
                unwrapped_args = torch.utils._pytree.tree_map(_unwrap, args)
                unwrapped_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
                return func(*unwrapped_args, **unwrapped_kwargs)
            if _has_zero(0) and len(args) > 1 and isinstance(args[1], ZeroTensor):
                meta_out = _infer_shape(func, args, kwargs)
                return _make_zero_like(meta_out)

        if func in (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar):
            if _has_zero(0) and len(args) > 1 and not isinstance(args[1], ZeroTensor):
                other = args[1]
                if isinstance(other, torch.Tensor):
                    return other.neg()
                unwrapped_args = torch.utils._pytree.tree_map(_unwrap, args)
                unwrapped_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
                return func(*unwrapped_args, **unwrapped_kwargs)
            if len(args) > 1 and isinstance(args[1], ZeroTensor) and not isinstance(args[0], ZeroTensor):
                other = args[0]
                if isinstance(other, torch.Tensor):
                    return other.clone()
                unwrapped_args = torch.utils._pytree.tree_map(_unwrap, args)
                unwrapped_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
                return func(*unwrapped_args, **unwrapped_kwargs)
            if _has_zero(0) and len(args) > 1 and isinstance(args[1], ZeroTensor):
                meta_out = _infer_shape(func, args, kwargs)
                return _make_zero_like(meta_out)

        # --- sum / mean over a zero tensor ---
        if func in (
            torch.ops.aten.sum.default,
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.mean.default,
            torch.ops.aten.mean.dim,
        ) and _has_zero(0):
            meta_out = _infer_shape(func, args, kwargs)
            return _make_zero_like(meta_out)

        # --- linear (addmm): bias + input @ weight.T ---
        if func == torch.ops.aten.addmm.default:
            # args: (bias, input, weight)
            if isinstance(args[1], ZeroTensor):
                # input is zero → result is just the bias
                return args[0].clone() if isinstance(args[0], torch.Tensor) else args[0]

        # --- cat/stack of all zeros ---
        if func in (torch.ops.aten.cat.default, torch.ops.aten.stack.default):
            tensors = args[0]
            if all(isinstance(t, ZeroTensor) for t in tensors):
                meta_out = _infer_shape(func, args, kwargs)
                return _make_zero_like(meta_out)

        # --- detach, alias ---
        if func in (
            torch.ops.aten.detach.default,
            torch.ops.aten.alias.default,
        ) and _has_zero(0):
            return args[0]

        # --- einsum with a zero arg ---
        if func == torch.ops.aten.einsum.default:
            equation = args[0]
            tensors = args[1]
            if any(isinstance(t, ZeroTensor) for t in tensors):
                meta_out = _infer_shape(func, args, kwargs)
                return _make_zero_like(meta_out)

        # --- fallback: materialize and compute normally ---
        unwrapped_args = torch.utils._pytree.tree_map(_unwrap, args)
        unwrapped_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
        return func(*unwrapped_args, **unwrapped_kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        with torch._C.DisableTorchFunctionSubclass():
            kwargs = kwargs or {}
            return super().__torch_function__(func, types, args, kwargs)
