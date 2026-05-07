"""Base LinearOp class and fundamental composition operators."""

import enum
from functools import reduce
from typing import Union

import torch
import warnings

from boundlab.sparse import coo
from boundlab.sparse.dim import Dim

from boundlab.sparse.coo import MultiCOOTensor, MultiCOOTensorSum, MultiCOOTensorSum
from boundlab.sparse.tn import Dense

class LinearOpFlags(enum.Flag):
    """Flags for LinearOps that can be used for optimization and simplification."""
    NONE = 0
    IS_NON_NEGATIVE = enum.auto()  # Output is guaranteed to be non-negative for non-negative input

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

    def __init__(self, tensor: Union[MultiCOOTensor, MultiCOOTensorSum], input_dims: list[Dim], output_dims: list[Dim], flags: LinearOpFlags = LinearOpFlags.NONE):
        """Initialize a LinearOp wrapper.

        Args:
            tensor: The underlying tensor representing the linear operator.
            input_dims: The dimensions of the input tensor.
            output_dims: The dimensions of the output tensor.
            flags: Flags indicating special properties of this LinearOp.
        """
        self.input_dims = [dim.clone() for dim in input_dims]
        self.output_dims = [dim.clone() for dim in output_dims]
        input_dims_map = {dim: self.input_dims[idx] for idx, dim in enumerate(input_dims)}
        output_dims_map = {dim: self.output_dims[idx] for idx, dim in enumerate(output_dims)}
        inner_dims = sorted(set(tensor.inner_dims) - set(input_dims) - set(output_dims))
        self.inner_dims = [dim.clone() for dim in inner_dims]
        inner_dims_map = {dim: self.inner_dims[idx] for idx, dim in enumerate(inner_dims)}

        self.tensor = tensor.replace_dims({**input_dims_map, **output_dims_map, **inner_dims_map})
        self.input_shape = torch.Size([dim.length for dim in input_dims])
        self.output_shape = torch.Size([dim.length for dim in output_dims])
        self.flags = flags
        self.name = None

    def __str__(self):
        return self.name if self.name else "{ " + str(self.tensor) + " }"

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
        assert x.shape == self.input_shape, f"Expected input shape {self.input_shape}, got {x.shape}."
        dense = self.tensor.tensordot(Dense(x, self.input_dims), self.input_dims).to_dense()
        return dense.expand(self.output_dims)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Apply the transposed linear function to an input tensor."""
        assert grad_output.shape == self.output_shape, f"Expected gradient output shape {self.output_shape}, got {grad_output.shape}."
        dense = self.tensor.tensordot(Dense(grad_output, self.output_dims), self.output_dims).to_dense()
        return dense.expand(self.input_dims)

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the original linear function to an input tensor, supporting additional trailing dimensions for batching."""
        assert x.shape[:len(self.input_shape)] == self.input_shape, f"Expected input shape starting with {self.input_shape}, got {x.shape}."
        additional_dims = [Dim(i, 1000 + i) for i in range(len(x.shape) - len(self.input_shape))]
        x_dense = Dense(x, self.input_dims + additional_dims)
        dense = self.tensor.tensordot(x_dense, self.input_dims).to_dense()
        return dense.expand(self.output_dims + additional_dims)


    def vbackward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Apply the transposed linear function to an input tensor, supporting additional leading dimensions for batching."""
        assert grad_output.shape[:len(self.output_shape)] == self.output_shape, f"Expected gradient output shape starting with {self.output_shape}, got {grad_output.shape}."
        additional_dims = [Dim(i, 1000 + i) for i in range(len(grad_output.shape) - len(self.output_shape))]
        grad_output_dense = Dense(grad_output, self.output_dims + additional_dims)
        dense = self.tensor.tensordot(grad_output_dense, self.output_dims).to_dense()
        return dense.expand(self.input_dims + additional_dims)

    def __mul__(self, other: float) -> "LinearOp":
        """Scale this LinearOp by a scalar factor."""
        if isinstance(other, (int, float)):
            return LinearOp(tensor=self.tensor * other, input_dims=self.input_dims, output_dims=self.output_dims, flags=self.flags)
        return NotImplemented

    def __rmul__(self, other: float) -> "LinearOp":
        """Scale this LinearOp by a scalar factor."""
        if isinstance(other, (int, float)):
            return LinearOp(tensor=self.tensor * other, input_dims=self.input_dims, output_dims=self.output_dims, flags=self.flags)
        return NotImplemented


    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose this LinearOp with another (self ∘ other)."""
        assert self.input_shape == other.output_shape, f"Cannot compose LinearOps with incompatible shapes: {self.input_shape} and {other.output_shape}."
        other_tensor = other.tensor
        other_tensor.replace_dims({a: b for a, b in zip(other.output_dims, self.input_dims)})
        tensor = coo.tensordot(self.tensor, other_tensor, self.input_dims)
        return LinearOp(tensor=tensor, input_dims=other.input_dims, output_dims=self.output_dims, flags=self.flags | other.flags)


    def __add__(self, other: "LinearOp") -> "LinearOp":
        """Add this LinearOp to another."""
        other_tensor = other.tensor
        other_tensor = other_tensor.replace_dims(
            {a: b for a, b in zip(other.input_dims, self.input_dims)}
            | {a: b for a, b in zip(other.output_dims, self.output_dims)}
        )
        if isinstance(other_tensor, MultiCOOTensor):
            other_tensor = MultiCOOTensorSum([other_tensor])
        if isinstance(self.tensor, MultiCOOTensor):
            self_tensor = MultiCOOTensorSum([self.tensor])

        return LinearOp(tensor=self.tensor + other_tensor, input_dims=self.input_dims, output_dims=self.output_dims, flags=self.flags & other.flags)

    
    def jacobian(self) -> torch.Tensor:
        """Return an explicit Jacobian tensor when efficiently available.

        Returns:
            A tensor with shape ``[*output_shape, *input_shape]`` if the
            concrete Jacobian can be produced directly. Returns
            ``NotImplemented`` for operators that only support implicit
            application.
        """
        # warnings.warn(f"LinearOp {self} does not implement jacobian method. Falling back to force_jacobian, which may be inefficient.", stacklevel=2)
        return self.tensor.to_dense().expand(self.output_dims + self.input_dims)
    
    def abs(self, p: Union[int, float] = 1) -> "LinearOp":
        """Return a LinearOp representing the element-wise absolute value of this LinearOp."""
        if self.flags & LinearOpFlags.IS_NON_NEGATIVE and p == 1:
            return self
        else:
            if p == 1:
                return LinearOp(tensor=self.tensor.apply_multiplicative(lambda x: x.abs()), input_dims=self.input_dims, output_dims=self.output_dims, flags=self.flags | LinearOpFlags.IS_NON_NEGATIVE)
            if p % 2 == 0:
                return LinearOp(tensor=self.tensor.apply_multiplicative(lambda x: x.pow(p)), input_dims=self.input_dims, output_dims=self.output_dims, flags=self.flags | LinearOpFlags.IS_NON_NEGATIVE)
            else:
                return LinearOp(tensor=self.tensor.apply_multiplicative(lambda x: x.abs().pow(p)), input_dims=self.input_dims, output_dims=self.output_dims, flags=self.flags | LinearOpFlags.IS_NON_NEGATIVE)

    def sum_input(self) -> "LinearOp":
        """Return a LinearOp representing the sum over all input dimensions of this LinearOp."""
        summed_tensor = self.tensor.sum(dims=self.input_dims)
        return LinearOp(tensor=summed_tensor, input_dims=[], output_dims=self.output_dims, flags=self.flags)
    
    def sum_output(self) -> "LinearOp":
        """Return a LinearOp representing the sum over all output dimensions of this LinearOp."""
        summed_tensor = self.tensor.sum(dims=self.output_dims)
        return LinearOp(tensor=summed_tensor, input_dims=self.input_dims, output_dims=[], flags=self.flags)

    def norm_input(self, p=1) -> "LinearOp":
        """Return a LinearOp that computes the norm over the input dimensions, if supported."""
        inv = 1 if p == 1 else 1/p
        return self.abs(p).sum_input().abs(p=inv)
    
    def norm_output(self, p=1) -> "LinearOp":
        """Return a LinearOp that computes the norm over the output dimensions, if supported."""
        inv = 1 if p == 1 else 1/p
        return self.abs(p).sum_output().abs(p=inv)
    
    def __neg__(self):
        """Return the negation of this LinearOp."""
        return (-1) * self

    def __sub__(self, other):
        """Return ``self - other`` as ``self + (-other)``."""
        return self + (-other)
    
    def __repr__(self):
        return str(self)
    