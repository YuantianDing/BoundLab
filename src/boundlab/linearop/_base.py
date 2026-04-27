"""Base LinearOp class and fundamental composition operators."""

import enum
from functools import reduce

import torch
import warnings

class LinearOpFlags(enum.Flag):
    """Flags for LinearOps that can be used for optimization and simplification."""
    NONE = 0
    IS_NON_NEGATIVE = enum.auto()  # Output is guaranteed to be non-negative for non-negative input
    IS_PURE_EXPANDING = enum.auto()  # Forward does not sum over input (e.g. broadcasting, repeat, pure reindex).
    IS_PURE_CONTRACTING = enum.auto()  # Forward does not duplicate input (e.g. sum, norm, pure reindex).

class LinearOp:
    r"""A base class for linear operators that can be applied to boundlab expressions.

    Subclasses should implement the forward and backward methods to define the
    linear transformation and its transpose, respectively.  LinearOps can be
    composed using matrix multiplication (@) and added together using addition (+).
    """

    input_shape: "torch.Size"
    """Expected input tensor shape."""
    output_shape: "torch.Size"
    """Computed output tensor shape."""

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size, flags: LinearOpFlags = LinearOpFlags.NONE):
        """Initialize a LinearOp wrapper.

        Args:
            input_shape: The expected shape of input tensors.
            output_shape: The expected shape of output tensors.
            flags: Flags indicating special properties of this LinearOp.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.flags = flags
        self.name = None

    def __str__(self):
        return self.name if self.name else f"<unknown linop {self.input_shape} -> {self.output_shape}>"

    def __call__(self, x):
        """Apply this LinearOp to an expression, returning a Linear."""
        from boundlab.expr import Expr
        if isinstance(x, Expr):
            from boundlab.expr._affine import AffineSum
            return AffineSum((self, x))
        elif isinstance(x, torch.Tensor):
            return self.forward(x)
        else:
            raise TypeError(f"LinearOp can only be applied to LinearOps or torch.Tensors, got {type(x)}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the original linear function to an input tensor."""
        raise NotImplementedError("Subclasses of LinearOp must implement the forward method.")

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Apply the transposed linear function to an input tensor."""
        raise NotImplementedError("Subclasses of LinearOp must implement the backward method.")

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the original linear function to an input tensor, supporting additional trailing dimensions for batching."""
        assert x.shape[:len(self.input_shape)] == self.input_shape, f"Expected input shape {self.input_shape}, got {x.shape}."
        orig_additional_dims = x.shape[len(self.input_shape):]
        x = x.reshape(*self.input_shape, -1)
        result = self._vmaped_forward(x)
        return result.reshape(*self.output_shape, *orig_additional_dims)
    
    def _vmaped_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.vmap(self.forward, in_dims=-1, out_dims=-1)(x)

    def vbackward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Apply the transposed linear function to an input tensor, supporting additional leading dimensions for batching."""
        assert grad_output.shape[-len(self.output_shape):] == self.output_shape, f"Expected gradient output shape {self.output_shape}, got {grad_output.shape}."
        orig_additional_dims = grad_output.shape[:-len(self.output_shape)]
        grad_output = grad_output.reshape(-1, *self.output_shape)
        result = self._vmaped_backward(grad_output)
        return result.reshape(*orig_additional_dims, *self.input_shape)

    def _vmaped_backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        return torch.vmap(self.backward)(grad_output)

    def __mul__(self, other: float) -> "LinearOp":
        """Scale this LinearOp by a scalar factor."""
        if isinstance(other, (int, float)):
            return ScalarOp(other, self.input_shape) @ self
        return NotImplemented

    def __rmul__(self, other: float) -> "LinearOp":
        """Scale this LinearOp by a scalar factor."""
        if isinstance(other, (int, float)):
            return ScalarOp(other, self.input_shape) @ self
        return NotImplemented


    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose this LinearOp with another (self ∘ other)."""
        return NotImplemented

    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose another LinearOp with this one (other ∘ self)."""
        if isinstance(other, LinearOp):
            # warnings.warn(f"Failed to fuse(@) LinearOps: {other} @ {self}, returning a ComposedOp. Consider implementing __matmul__ for these LinearOp types for better performance.", stacklevel=2)
            return ComposedOp(other, self)
        return NotImplemented

    def __add__(self, other: "LinearOp") -> "LinearOp":
        """Add this LinearOp to another."""
        if isinstance(other, LinearOp):
            # warnings.warn(f"Failed to fuse(+) LinearOps: {self} + {other}, returning a SumOp. Consider implementing __add__ for these LinearOp types for better performance.", stacklevel=2)
            return SumOp(self, other)
        return NotImplemented

    def __radd__(self, other: "LinearOp") -> "LinearOp":
        """Add another LinearOp to this one."""
        if isinstance(other, LinearOp):
            return SumOp(other, self)
        return NotImplemented
    
    def einsum_op(self) -> "EinsumOp":
        """Materialize this LinearOp as an explicit Jacobian tensor.

        Returns:
            A tensor with shape ``[*output_shape, *input_shape]`` representing
            the Jacobian of this LinearOp.
        Notes:
            This may be expensive in time and memory and is mainly intended for
            debugging, validation, or rare paths that require explicit Jacobians.
        """
        from boundlab.linearop._einsum import EinsumOp
        if isinstance(self, EinsumOp):
            return self
        jac = self.jacobian()
            
        return EinsumOp.from_full(jac, len(self.input_shape))
    
    def jacobian(self) -> torch.Tensor:
        """Return an explicit Jacobian tensor when efficiently available.

        Returns:
            A tensor with shape ``[*output_shape, *input_shape]`` if the
            concrete Jacobian can be produced directly. Returns
            ``NotImplemented`` for operators that only support implicit
            application.
        """
        # warnings.warn(f"LinearOp {self} does not implement jacobian method. Falling back to force_jacobian, which may be inefficient.", stacklevel=2)
        return self.force_jacobian()
    
    def abs(self) -> "LinearOp":
        """Return a LinearOp representing the element-wise absolute value of this LinearOp."""
        if self.flags & LinearOpFlags.IS_NON_NEGATIVE:
            return self
        raise NotImplementedError(f"LinearOp {self} does not implement abs method, and is not guaranteed to be non-negative.")
    
    def norm_input(self, p=1) -> "LinearOp":
        """Return a LinearOp that computes the norm over the input dimensions, if supported."""
        from boundlab.linearop._einsum import EinsumOp
        if self.flags & LinearOpFlags.IS_PURE_CONTRACTING and self.flags & LinearOpFlags.IS_NON_NEGATIVE:
            tensor = self.forward(torch.ones(self.input_shape))
            return EinsumOp.from_full(tensor, 0, name=f"norm_input(p={p}) of {self}" if self.name else None)
        raise NotImplementedError(f"LinearOp {self} does not implement norm_input method.")
    
    def norm_output(self, p=1) -> "LinearOp":
        """Return a LinearOp that computes the norm over the output dimensions, if supported."""
        if self.flags & LinearOpFlags.IS_PURE_EXPANDING and self.flags & LinearOpFlags.IS_NON_NEGATIVE:
            tensor = self.backward(torch.ones(self.output_shape))
            from boundlab.linearop._einsum import EinsumOp
            return EinsumOp.from_full(tensor, len(self.input_shape), name=f"norm_output(p={p}) of {self}" if self.name else None)
        raise NotImplementedError(f"LinearOp {self} does not implement norm_output method.")
    
    def __neg__(self):
        """Return the negation of this LinearOp."""
        return (-1) * self

    def __sub__(self, other):
        """Return ``self - other`` as ``self + (-other)``."""
        if isinstance(other, LinearOp):
            return self + (-other)
        return NotImplemented
    
    def __repr__(self):
        return str(self)
    
    def force_jacobian(self):
        """Materialize Jacobian via batched forward/backward application.

        This fallback constructs an identity basis and applies either
        :meth:`vforward` or :meth:`vbackward` depending on whether the input or
        output side is smaller.

        Returns:
            A dense Jacobian tensor with shape ``[*output_shape, *input_shape]``.

        Notes:
            This may be expensive in time and memory and is mainly intended for
            debugging, validation, or rare paths that require explicit Jacobians.
        """
        input_numel = self.input_shape.numel()
        output_numel = self.output_shape.numel()
        if input_numel < output_numel:
            jacobian = torch.eye(input_numel).reshape(*self.input_shape, *self.input_shape)
            jacobian = self.vforward(jacobian)
            assert jacobian.shape == (self.output_shape + self.input_shape), f"Expected Jacobian shape {self.output_shape + self.input_shape}, got {jacobian.shape}."
            return jacobian
        else:
            shape = torch.Size(self.output_shape + self.output_shape)
            if len(shape) > 0:
                jacobian = torch.eye(output_numel).reshape(*shape)
            else:
                jacobian = torch.tensor(1.0)
            jacobian = self.vbackward(jacobian)
            assert jacobian.shape == self.output_shape + self.input_shape, f"Expected Jacobian shape {self.output_shape + self.input_shape}, got {jacobian.shape}. {self}"
            return jacobian
    
    def jacobian_scatter(self, src: torch.Tensor) -> torch.Tensor:
        """Add this operator's Jacobian contribution into an existing tensor.

        Args:
            src: A tensor with Jacobian layout ``[*output_shape, *input_shape]``
                that acts as the accumulation buffer.

        Returns:
            A tensor with the same shape as ``src`` containing
            ``src + jacobian(self)``.

        Notes:
            Subclasses may override this to implement structured/sparse updates
            without materializing the full Jacobian first.
        """
        return NotImplemented

class ComposedOp(LinearOp):
    """Composition of two LinearOps: ``(outer ∘ inner)(x) = outer(inner(x))``."""

    def __init__(self, *ops: LinearOp):
        self.ops = []
        for op in ops:
            if isinstance(op, ComposedOp):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)
        inner = self.ops[-1]
        outer = self.ops[0]
        for i in range(len(self.ops) - 1):
            assert self.ops[i].input_shape == self.ops[i + 1].output_shape, \
                f"Shape mismatch in ComposedOp: {self.ops[i].input_shape}->{self.ops[i].output_shape} vs {self.ops[i + 1].input_shape}->{self.ops[i + 1].output_shape}"
        super().__init__(inner.input_shape, outer.output_shape)
        if all(op.flags & LinearOpFlags.IS_NON_NEGATIVE for op in self.ops):
            self.flags |= LinearOpFlags.IS_NON_NEGATIVE
        if all(op.flags & LinearOpFlags.IS_PURE_EXPANDING for op in self.ops):
            self.flags |= LinearOpFlags.IS_PURE_EXPANDING
        if all(op.flags & LinearOpFlags.IS_PURE_CONTRACTING for op in self.ops):
            self.flags |= LinearOpFlags.IS_PURE_CONTRACTING

    def forward(self, x):
        for op in reversed(self.ops):
            x = op.forward(x)
        return x

    def backward(self, grad):
        for op in self.ops:
            grad = op.backward(grad)
        return grad

    def vforward(self, x):
        for op in reversed(self.ops):
            x = op.vforward(x)
        return x
    
    def vbackward(self, grad):
        for op in self.ops:
            grad = op.vbackward(grad)
        return grad

    def __str__(self):
        return "(" + " ∘ ".join(str(op) for op in self.ops) + ")"
    
    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        if isinstance(other, ComposedOp):
            assert self.input_shape == other.output_shape, \
                f"Shape mismatch in ComposedOp: {self.input_shape} vs {other.output_shape}"
            return ComposedOp(*(self.ops + other.ops))
        if isinstance(other, LinearOp):
            for i in range(len(self.ops) - 1, -1, -1):
                other = self.ops[i] @ other
                if isinstance(other, ComposedOp):
                    return ComposedOp(*(self.ops[:i] + other.ops))
            return other
        return NotImplemented

    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        if isinstance(other, ComposedOp):
            assert other.input_shape == self.output_shape, \
                f"Shape mismatch in ComposedOp: {other.input_shape} vs {self.output_shape}"
            return ComposedOp(*(other.ops + self.ops))
        if isinstance(other, LinearOp):
            for i, op in enumerate(self.ops):
                other = other @ op
                if isinstance(other, ComposedOp):
                    return ComposedOp(*(other.ops + self.ops[i + 1:]))
            return other
        return super().__rmatmul__(other)
    
    def jacobian(self):
        if self.input_shape.numel() >= self.output_shape.numel():
            jac = self.ops[0].jacobian()
            for op in self.ops[1:]:
                jac = op.vbackward(jac)
            return jac
        elif self.output_shape.numel() > self.input_shape.numel():
            jac = self.ops[-1].jacobian()
            for op in reversed(self.ops[:-1]):
                jac = op.vforward(jac)
            return jac
    
    def purify(self) -> bool:
        i = 0
        while self.ops[i].flags & LinearOpFlags.IS_PURE_EXPANDING:
            i += 1
            if i == len(self.ops):
                break
        
        j = len(self.ops) - 1
        while self.ops[j].flags & LinearOpFlags.IS_PURE_CONTRACTING:
            j -= 1
            if j < 0:
                break
        
        expanding_ops = self.ops[:i]
        contracting_ops = self.ops[j + 1:]
        middle_ops = self.ops[i:j + 1]
        if len(middle_ops) > 1:
            return False
        elif len(middle_ops) == 1:
            if isinstance(middle_ops[0], SumOp):
                return middle_ops[0].purify()
            return True
        else:
            return True
                
    
    def norm_input(self, p=1):
        assert p == 1
        if self.purify():
            op = self.ops[-1].norm_input(p)
            for other_op in reversed(self.ops[:-1]):
                op = other_op.abs() @ op
            return op
        else:
            return super().norm_input(p)
    
    def norm_output(self, p=1):
        assert p == 1
        if self.purify():
            op = self.ops[0].norm_output(p)
            for other_op in self.ops[1:]:
                op = op @ other_op.abs()
            return op
        else:
            return super().norm_output(p)

    def purify_with(self, other):
        """Materialize as EinsumOp and delegate to EinsumOp.purify_with."""
        return self.einsum_op().purify_with(other)





class SumOp(LinearOp):
    """Sum of two LinearOps: ``(a + b)(x) = a(x) + b(x)``."""

    def __init__(self, *ops: LinearOp):
        self.ops = []
        for op in ops:
            if isinstance(op, SumOp):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)
        inner = self.ops[0]
        outer = self.ops[0]
        assert all(op.input_shape == inner.input_shape and op.output_shape == inner.output_shape for op in self.ops), \
            f"Shape mismatch in SumOp: {self.ops} {[op.input_shape for op in self.ops]} -> {[op.output_shape for op in self.ops]}"
        super().__init__(inner.input_shape, outer.output_shape)

        if all(op.flags & LinearOpFlags.IS_NON_NEGATIVE for op in self.ops):
            self.flags |= LinearOpFlags.IS_NON_NEGATIVE
        if all(op.flags & LinearOpFlags.IS_PURE_EXPANDING for op in self.ops):
            self.flags |= LinearOpFlags.IS_PURE_EXPANDING
        if all(op.flags & LinearOpFlags.IS_PURE_CONTRACTING for op in self.ops):
            self.flags |= LinearOpFlags.IS_PURE_CONTRACTING

        self.purified = False

    def forward(self, x):
        return sum(op.forward(x) for op in self.ops)
    
    def vforward(self, x):
        return sum(op.vforward(x) for op in self.ops)

    def backward(self, grad):
        return sum(op.backward(grad) for op in self.ops)

    def vbackward(self, grad):
        return sum(op.vbackward(grad) for op in self.ops)

    def __str__(self):
        return "(" + " + ".join(str(op) for op in self.ops) + ")"
    
    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        from boundlab.linearop._einsum import EinsumOp
        ops = [op @ other for op in self.ops]
        if not any(isinstance(op, ComposedOp) for op in ops):
            return sum(ops, start=ZeroOp(other.input_shape, self.output_shape))
        return NotImplemented

    def __rmatmul__(self, other):
        from boundlab.linearop._einsum import EinsumOp
        ops = [other @ op for op in self.ops]
        if not any(isinstance(op, ComposedOp) for op in ops):
            return sum(ops, start=ZeroOp(self.input_shape, other.output_shape))
        return super().__rmatmul__(other)
    
    def __add__(self, other: "LinearOp") -> "LinearOp":
        if isinstance(other, LinearOp):
            other = SumOp(other)
        if isinstance(other, SumOp):
            assert self.input_shape == other.input_shape and self.output_shape == other.output_shape
            result_self_ops = self.ops.copy()
            result_other_ops = []
            for op in other.ops:
                for i, _ in enumerate(result_self_ops):
                    o = result_self_ops[i] + op
                    if not isinstance(o, SumOp):
                        result_self_ops[i] = o
                        break
                else:
                    result_other_ops.append(op)
            result = result_self_ops + result_other_ops
            if len(result) == 1:
                return result[0]
            else:
                return SumOp(*result)
        return NotImplemented
    
    def jacobian(self):
        jacs = [op.jacobian() for op in self.ops]
        return sum(jacs)
    
    def norm_input(self, p=1) -> "LinearOp":
        """Return a LinearOp that computes the norm over the input dimensions, if supported."""
        if not self.purified:
            self.purify()
        
        return sum((op.norm_input(p) for op in self.ops), start=ZeroOp(self.input_shape, torch.Size(())))
    
    def norm_output(self, p=1) -> "LinearOp":
        """Return a LinearOp that computes the norm over the output dimensions, if supported."""
        if not self.purified:
            self.purify()
        
        return sum((op.norm_output(p) for op in self.ops), start=ZeroOp(torch.Size(()), self.output_shape))
    
    
    def purify(self) -> bool:
        if self.purified:
            return self
        
        i = 0
        while i < len(self.ops):
            for j in range(0, i):
                result = self.ops[j].purify_with(self.ops[i])
                if isinstance(result, tuple) and len(result) == 2:
                    self.ops[j], self.ops[i] = result
                elif isinstance(result, LinearOp):
                    self.ops[j] = result
                    del self.ops[i]
                    break
                else:
                    raise NotImplementedError(f"SumOp purify_with not implemented for {type(self.ops[j])} and {type(self.ops[i])}")
            else:
                i += 1

        self.purified = True
        return self.purified


class ScalarOp(LinearOp):
    """A LinearOp that scales its input by a scalar factor."""

    def __init__(self, scalar: float, input_shape: torch.Size, name=None):
        self.scalar = scalar
        self.name = name
        super().__init__(input_shape, input_shape)

        if scalar >= 0:
            self.flags |= LinearOpFlags.IS_NON_NEGATIVE
        self.flags |= LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scalar == 1.0:
            return x
        return self.scalar * x

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self.scalar == 1.0:
            return grad_output
        return self.scalar * grad_output

    def vforward(self, x):
        if self.scalar == 1.0:
            return x
        return self.scalar * x

    def vbackward(self, grad_output):
        if self.scalar == 1.0:
            return grad_output
        return self.scalar * grad_output

    def __str__(self):
        return self.name if self.name is not None else f"{self.scalar}"

    def __mul__(self, other: float) -> "ScalarOp":
        if isinstance(other, (int, float)):
            return ScalarOp(self.scalar * other, self.input_shape)
        return NotImplemented

    def __rmul__(self, other: float) -> "ScalarOp":
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, ScalarOp):
            return ScalarOp(self.scalar * other.scalar, other.input_shape)
        if isinstance(other, LinearOp):
            if self.scalar == 1.0:
                return other
            else:
                return NotImplemented
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, LinearOp):
            if self.scalar == 1.0:
                return other
            else:
                return super().__rmatmul__(other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, ScalarOp):
            return ScalarOp(self.scalar + other.scalar, self.input_shape)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, ScalarOp):
            return other.__add__(self)
        return super().__radd__(other)
    
    def purify_with(self, other):
        return NotImplemented

    def is_identity(self) -> bool:
        return self.scalar == 1.0
    
    def abs(self):
        return ScalarOp(abs(self.scalar), self.input_shape, name=f"|{self}|" if self.name else None)

class ZeroOp(LinearOp):
    """A LinearOp that always returns zero, regardless of input."""

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size, name=None):
        if name is not None:
            self.name = name
        super().__init__(
            input_shape,
            output_shape,
            flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING,
        )

    def forward(self, x):
        return torch.zeros(self.output_shape, dtype=x.dtype, device=x.device)

    def backward(self, grad):
        return torch.zeros(self.input_shape, dtype=grad.dtype, device=grad.device)

    def vforward(self, x):
        extra = x.shape[len(self.input_shape):]
        return torch.zeros(*self.output_shape, *extra, dtype=x.dtype, device=x.device)

    def vbackward(self, grad):
        extra = grad.shape[:-len(self.output_shape)]
        return torch.zeros(*extra, *self.input_shape, dtype=grad.dtype, device=grad.device)

    def __str__(self):
        return self.name if self.name else "0"

    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        if isinstance(other, LinearOp):
            return ZeroOp(other.input_shape, self.output_shape)
        return NotImplemented

    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        if isinstance(other, LinearOp):
            return ZeroOp(self.input_shape, other.output_shape)
        return NotImplemented

    def __add__(self, other: "LinearOp") -> "LinearOp":
        if isinstance(other, LinearOp):
            return other
        return NotImplemented

    def __radd__(self, other: "LinearOp") -> "LinearOp":
        if isinstance(other, LinearOp):
            return other
        return NotImplemented
    
    def jacobian(self):
        return torch.zeros(self.output_shape + self.input_shape)
