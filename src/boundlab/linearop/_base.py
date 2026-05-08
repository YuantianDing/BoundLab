"""Base LinearOp class and fundamental composition operators."""

import enum
from functools import reduce
from typing import Union

import torch
import warnings

from boundlab.sparse import coo
from boundlab.sparse.dim import Dim

from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum, MultiCOOTensorSum
from boundlab.sparse.tn import Dense

class LinearOpFlags(enum.Flag):
    """Flags for LinearOps that can be used for optimization and simplification."""
    NONE = 0
    IS_NON_NEGATIVE = enum.auto()  # Output is guaranteed to be non-negative for non-negative input
import os
DEBUG_LINEAR_OP = os.environ.get("DEBUG_LINEAR_OP", "0") == "1"
if DEBUG_LINEAR_OP:
    print("DEBUG_LINEAR_OP is enabled. This may cause significant slowdown. Only enable this if you are debugging LinearOps.")

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

    def __init__(self, tensor: Union[MultiCOOTensor, MultiCOOTensorSum], input_dims: list[Dim], output_dims: list[Dim], flags: LinearOpFlags = LinearOpFlags.NONE, debug_jacobian: torch.Tensor | None = None):
        """Initialize a LinearOp wrapper.

        Args:
            tensor: The underlying tensor representing the linear operator.
            input_dims: The dimensions of the input tensor.
            output_dims: The dimensions of the output tensor.
            flags: Flags indicating special properties of this LinearOp.
            debug_jacobian: A tensor for debugging the Jacobian of this LinearOp.
        """
        self.input_dims = [dim.clone(name=f"i{idx}") for idx, dim in enumerate(input_dims)]
        self.output_dims = [dim.clone(name=f"o{idx}") for idx, dim in enumerate(output_dims)]
        input_dims_map = {dim: self.input_dims[idx] for idx, dim in enumerate(input_dims)}
        output_dims_map = {dim: self.output_dims[idx] for idx, dim in enumerate(output_dims)}
        inner_dims = sorted(set(tensor.inner_dims) - set(input_dims) - set(output_dims))
        self.inner_dims = [dim.clone(name=f"k{idx}") for idx, dim in enumerate(inner_dims)]
        inner_dims_map = {dim: self.inner_dims[idx] for idx, dim in enumerate(inner_dims)}

        self.tensor = tensor.replace_dims({**input_dims_map, **output_dims_map, **inner_dims_map})
        self.input_shape = torch.Size([dim.length for dim in input_dims])
        self.output_shape = torch.Size([dim.length for dim in output_dims])
        self.flags = flags
        self.name = None
        if DEBUG_LINEAR_OP:
            assert debug_jacobian is not None
            self.debug_jacobian = Dense(debug_jacobian, self.output_dims + self.input_dims)
        else:
            self.debug_jacobian = None

    def __str__(self):
        return self.name if self.name else "{" + str(self.tensor) + "}"

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
        x_dense = Dense(x, self.input_dims)
        dense = self.tensor.tensordot(x_dense, self.input_dims).to_dense()
        if DEBUG_LINEAR_OP:
            assert self.debug_jacobian.tensordot(x_dense, self.input_dims).allclose(dense)
        return dense.expand(self.output_dims)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Apply the transposed linear function to an input tensor."""
        assert grad_output.shape == self.output_shape, f"Expected gradient output shape {self.output_shape}, got {grad_output.shape}."
        grad_output_dense = Dense(grad_output, self.output_dims)
        dense = self.tensor.tensordot(grad_output_dense, self.output_dims).to_dense()
        if DEBUG_LINEAR_OP:
            assert self.debug_jacobian.tensordot(grad_output_dense, self.output_dims).allclose(dense)
        return dense.expand(self.input_dims)

    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the original linear function to an input tensor, supporting additional trailing dimensions for batching."""
        assert x.shape[:len(self.input_shape)] == self.input_shape, f"Expected input shape starting with {self.input_shape}, got {x.shape}."
        additional_dims = [
            Dim(length=int(x.shape[len(self.input_shape) + i]), ordering=2000.0 + i, name=f"b{i}")
            for i in range(len(x.shape) - len(self.input_shape))
        ]
        x_dense = Dense(x, self.input_dims + additional_dims)
        dense = self.tensor.tensordot(x_dense, self.input_dims).to_dense()
        if DEBUG_LINEAR_OP :
            assert self.debug_jacobian.tensordot(x_dense, self.input_dims).allclose(dense)
        return dense.expand(self.output_dims + additional_dims)


    def vbackward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Apply the transposed linear function to an input tensor, supporting additional leading dimensions for batching."""
        batch_shape = (
            torch.Size(grad_output.shape[:-len(self.output_shape)])
            if len(self.output_shape) > 0
            else torch.Size(grad_output.shape)
        )
        event_shape = torch.Size(grad_output.shape[len(batch_shape):])
        assert event_shape == self.output_shape, f"Expected trailing output shape {self.output_shape}, got {grad_output.shape}."
        additional_dims = [
            Dim(length=int(batch_shape[i]), ordering=2000.0 + i, name=f"b{i}")
            for i in range(len(batch_shape))
        ]
        grad_output_dense = Dense(grad_output, additional_dims + self.output_dims)
        dense = self.tensor.tensordot(grad_output_dense, self.output_dims).to_dense()
        if DEBUG_LINEAR_OP:
            assert self.debug_jacobian.tensordot(grad_output_dense, self.output_dims).allclose(dense)
        return dense.expand(additional_dims + self.input_dims)

    def __mul__(self, other: float) -> "LinearOp":
        """Scale this LinearOp by a scalar factor."""
        if isinstance(other, (int, float)):
            flags = self.flags
            if other < 0:
                flags &= ~LinearOpFlags.IS_NON_NEGATIVE
            debug_jacobian = self.debug_jacobian.expand(self.output_dims + self.input_dims) * other if self.debug_jacobian is not None else None
            return LinearOp(
                tensor=self.tensor * other,
                input_dims=self.input_dims,
                output_dims=self.output_dims,
                flags=flags,
                debug_jacobian=debug_jacobian,
            )
        return NotImplemented

    def __rmul__(self, other: float) -> "LinearOp":
        """Scale this LinearOp by a scalar factor."""
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented


    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose this LinearOp with another (self ∘ other)."""
        assert self.input_shape == other.output_shape, f"Cannot compose LinearOps with incompatible shapes: {self.input_shape} and {other.output_shape}."
        output_dim_map = {a: b for a, b in zip(other.output_dims, self.input_dims)}
        other_tensor = other.tensor
        other_tensor = other_tensor.replace_dims(output_dim_map)
        tensor = coo.tensordot(self.tensor, other_tensor, self.input_dims)
        debug_jacobian = None
        if DEBUG_LINEAR_OP:
            other_debug_jacobian = other.debug_jacobian.replace_dims(output_dim_map)
            jac = self.debug_jacobian.tensordot(other_debug_jacobian, self.input_dims)
            assert jac.allclose(tensor.to_dense())
            debug_jacobian = jac.expand(self.output_dims + other.input_dims)

        return LinearOp(tensor=tensor, input_dims=other.input_dims, output_dims=self.output_dims, flags=self.flags & other.flags, debug_jacobian=debug_jacobian)


    def __add__(self, other: "LinearOp") -> "LinearOp":
        """Add this LinearOp to another."""
        input_dim_map = {a: b for a, b in zip(other.input_dims, self.input_dims)}
        output_dim_map = {a: b for a, b in zip(other.output_dims, self.output_dims)}
        other_tensor = other.tensor
        other_tensor = other_tensor.replace_dims(input_dim_map | output_dim_map)
        self_tensor = self.tensor
        if isinstance(other_tensor, MultiCOOTensor):
            other_tensor = MultiCOOTensorSum([other_tensor])
        if isinstance(self.tensor, MultiCOOTensor):
            self_tensor = MultiCOOTensorSum([self.tensor])

        debug_jacobian = None
        if DEBUG_LINEAR_OP:
            other_debug_jacobian = other.debug_jacobian.replace_dims(input_dim_map | output_dim_map)
            jac = self.debug_jacobian + other_debug_jacobian
            assert jac.allclose((self_tensor + other_tensor).to_dense())
            debug_jacobian = jac.expand(self.output_dims + self.input_dims)

        return LinearOp(tensor=self_tensor + other_tensor, input_dims=self.input_dims, output_dims=self.output_dims, flags=self.flags & other.flags, debug_jacobian=debug_jacobian)

    
    def jacobian(self) -> torch.Tensor:
        """Return an explicit Jacobian tensor when efficiently available.

        Returns:
            A tensor with shape ``[*output_shape, *input_shape]`` if the
            concrete Jacobian can be produced directly. Returns
            ``NotImplemented`` for operators that only support implicit
            application.
        """
        # warnings.warn(f"LinearOp {self} does not implement jacobian method. Falling back to force_jacobian, which may be inefficient.", stacklevel=2)
        dense = self.tensor.to_dense()
        if DEBUG_LINEAR_OP and self.debug_jacobian is not None:
            print("Check")
            assert self.debug_jacobian.allclose(dense)
        return dense.expand(self.output_dims + self.input_dims)
    
    def abs(self, p: Union[int, float] = 1) -> "LinearOp":
        """Return a LinearOp representing the element-wise absolute value of this LinearOp."""
        if LinearOpFlags.IS_NON_NEGATIVE in self.flags and p == 1:
            return self
        else:
            if p == 1:
                result = self.apply_multiplicative(lambda x: x.abs())
            elif p % 2 == 0:
                result = self.apply_multiplicative(lambda x: x.abs().pow(p))
            else:
                result = self.apply_multiplicative(lambda x: x.abs().pow(p) * x.sign())
            result.flags |= (self.flags & LinearOpFlags.IS_NON_NEGATIVE)
            return result
            
    def apply_multiplicative(self, func) -> "LinearOp":
        """Return a LinearOp representing the element-wise application of a function to this LinearOp."""
        debug_jacobian = self.debug_jacobian.apply(func).expand(self.output_dims + self.input_dims) if self.debug_jacobian is not None else None
        return LinearOp(tensor=self.tensor.apply_multiplicative(func), input_dims=self.input_dims, output_dims=self.output_dims, flags=self.flags, debug_jacobian=debug_jacobian)

    def sum_input(self) -> "LinearOp":
        """Return a LinearOp representing the sum over all input dimensions of this LinearOp."""
        summed_tensor = self.tensor.sum(dims=self.input_dims)
        debug_jacobian = self.debug_jacobian.sum(self.input_dims).expand(self.output_dims) if self.debug_jacobian is not None else None
        if DEBUG_LINEAR_OP and self.debug_jacobian is not None:
            assert self.debug_jacobian.sum(self.input_dims).allclose(summed_tensor.to_dense()), f"{self} {summed_tensor} {self.debug_jacobian.sum(self.input_dims)} {summed_tensor.to_dense()}"
        return LinearOp(tensor=summed_tensor, input_dims=[], output_dims=self.output_dims, flags=self.flags, debug_jacobian=debug_jacobian)
    
    def sum_output(self) -> "LinearOp":
        """Return a LinearOp representing the sum over all output dimensions of this LinearOp."""
        summed_tensor = self.tensor.sum(dims=self.output_dims)
        debug_jacobian = self.debug_jacobian.sum(self.output_dims).expand(self.input_dims) if self.debug_jacobian is not None else None
        if DEBUG_LINEAR_OP and self.debug_jacobian is not None:
            assert self.debug_jacobian.sum(self.output_dims).allclose(summed_tensor.to_dense()), f"{self} {summed_tensor} {self.debug_jacobian.sum(self.output_dims)} {summed_tensor.to_dense()}"
        return LinearOp(tensor=summed_tensor, input_dims=self.input_dims, output_dims=[], flags=self.flags, debug_jacobian=debug_jacobian)

    def norm_input(self, p=1) -> "LinearOp":
        """Return a LinearOp that computes the norm over the input dimensions, if supported."""
        tensor = self.tensor
        if isinstance(self.tensor, MultiCOOTensorSum) and len(self.tensor.terms) == 1:
            tensor = self.tensor.terms[0]
        if isinstance(tensor, MultiCOOTensor):
            result = tensor.norm(p=p, dims=self.input_dims)
            debug_jacobian = None
            if DEBUG_LINEAR_OP:
                jac = self.debug_jacobian.norm(p=p, dims=self.input_dims)
                assert jac.allclose(result.to_dense()), f"{self} {result} {jac} {result.to_dense()}"
                debug_jacobian = jac.expand(self.output_dims)
            return LinearOp(
                tensor=result,
                input_dims=[],
                output_dims=self.output_dims,
                flags=self.flags | LinearOpFlags.IS_NON_NEGATIVE,
                debug_jacobian=debug_jacobian,
            )
        inv = 1 if p == 1 else 1/p
        return self.abs(p).sum_input().abs(p=inv)
    
    def norm_output(self, p=1) -> "LinearOp":
        """Return a LinearOp that computes the norm over the output dimensions, if supported."""
        tensor = self.tensor
        if isinstance(self.tensor, MultiCOOTensorSum) and len(self.tensor.terms) == 1:
            tensor = self.tensor.terms[0]
        if isinstance(tensor, MultiCOOTensor):
            result = tensor.norm(p=p, dims=self.output_dims)

            debug_jacobian = None
            if DEBUG_LINEAR_OP:
                jac = self.debug_jacobian.norm(p=p, dims=self.output_dims)
                assert jac.allclose(result.to_dense())
                debug_jacobian = jac.expand(self.input_dims)
            
            return LinearOp(
                tensor=result,
                input_dims=self.input_dims,
                output_dims=[],
                flags=self.flags | LinearOpFlags.IS_NON_NEGATIVE,
                debug_jacobian=debug_jacobian,
            )
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
    
