"""Base LinearOp class and fundamental composition operators."""

import torch


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

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        """Initialize a LinearOp wrapper.

        Args:
            input_shape: The expected shape of input tensors.
            output_shape: The expected shape of output tensors.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __str__(self):
        return self.name if hasattr(self, "name") else f"<unknown linop {self.input_shape} -> {self.output_shape}>"

    def __call__(self, x):
        """Apply this LinearOp to an expression, returning a Linear."""
        from boundlab.expr import Expr
        if isinstance(x, Expr):
            from boundlab.expr._linear import AffineSum
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
        result = torch.vmap(self.forward, in_dims=-1, out_dims=-1)(x)
        return result.reshape(*self.output_shape, *orig_additional_dims)

    def vbackward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Apply the transposed linear function to an input tensor, supporting additional leading dimensions for batching."""
        assert grad_output.shape[-len(self.output_shape):] == self.output_shape, f"Expected gradient output shape {self.output_shape}, got {grad_output.shape}."
        orig_additional_dims = grad_output.shape[:-len(self.output_shape)]
        grad_output = grad_output.reshape(-1, *self.output_shape)
        result = torch.vmap(self.backward)(grad_output)
        return result.reshape(*orig_additional_dims, *self.input_shape)
    
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
        if isinstance(other, LinearOp):
            return ComposedOp(self, other)
        return NotImplemented

    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose another LinearOp with this one (other ∘ self)."""
        if isinstance(other, LinearOp):
            return ComposedOp(other, self)
        return NotImplemented

    def __add__(self, other: "LinearOp") -> "LinearOp":
        """Add this LinearOp to another."""
        if isinstance(other, LinearOp):
            return SumOp(self, other)
        return NotImplemented

    def __radd__(self, other: "LinearOp") -> "LinearOp":
        """Add another LinearOp to this one."""
        if isinstance(other, LinearOp):
            return SumOp(other, self)
        return NotImplemented


class ComposedOp(LinearOp):
    """Composition of two LinearOps: ``(outer ∘ inner)(x) = outer(inner(x))``."""

    def __init__(self, outer: LinearOp, inner: LinearOp):
        self.outer = outer
        self.inner = inner
        super().__init__(inner.input_shape, outer.output_shape)

    def forward(self, x):
        return self.outer.forward(self.inner.forward(x))

    def backward(self, grad):
        return self.inner.backward(self.outer.backward(grad))

    def __str__(self):
        return f"({self.outer} ∘ {self.inner})"


class SumOp(LinearOp):
    """Sum of two LinearOps: ``(a + b)(x) = a(x) + b(x)``."""

    def __init__(self, a: LinearOp, b: LinearOp):
        self.a = a
        self.b = b
        assert a.input_shape == b.input_shape and a.output_shape == b.output_shape, \
            f"Shape mismatch in SumOp: {a.input_shape}->{a.output_shape} vs {b.input_shape}->{b.output_shape}"
        super().__init__(a.input_shape, a.output_shape)

    def forward(self, x):
        return self.a.forward(x) + self.b.forward(x)

    def backward(self, grad):
        return self.a.backward(grad) + self.b.backward(grad)

    def __str__(self):
        return f"({self.a} + {self.b})"


class ScalarOp(LinearOp):
    """A LinearOp that scales its input by a scalar factor."""

    def __init__(self, scalar: float, input_shape: torch.Size, name=None):
        self.scalar = scalar
        if name is not None:
            self.name = name
        super().__init__(input_shape, input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scalar * x

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        return self.scalar * grad_output

    def vforward(self, x):
        return self.scalar * x

    def vbackward(self, grad_output):
        return self.scalar * grad_output

    def __str__(self):
        return self.name if hasattr(self, "name") else f"{self.scalar}"

    def __mul__(self, other: float) -> "ScalarOp":
        if isinstance(other, (int, float)):
            return ScalarOp(self.scalar * other, self.input_shape)
        return NotImplemented

    def __rmul__(self, other: float) -> "ScalarOp":
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, LinearOp):
            if self.scalar == 1.0:
                return other
            else:
                return other * self.scalar
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, LinearOp):
            if self.scalar == 1.0:
                return other
            else:
                return other * self.scalar
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, ScalarOp):
            return ScalarOp(self.scalar + other.scalar, self.input_shape)
        return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, ScalarOp):
            return other.__add__(self)
        return super().__radd__(other)

    def is_identity(self) -> bool:
        return self.scalar == 1.0


class ZeroOp(LinearOp):
    """A LinearOp that always returns zero, regardless of input."""

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size, name=None):
        if name is not None:
            self.name = name
        super().__init__(input_shape, output_shape)

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
        return self.name if hasattr(self, "name") else "0"

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
