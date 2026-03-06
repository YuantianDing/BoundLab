r"""Linear Transformations with Constant Tensors

This module defines expression classes for linear operations between
constant tensors and symbolic expressions.
"""

from typing import Literal

import torch

from boundlab.expr._core import Expr, ExprFlags
from boundlab.utils import eye_of


class ConstTensorDot[ChildT: Expr](Expr):
    child: ChildT
    value: torch.Tensor
    dims: int
    name: str | None
    r"""Tensor contraction between a constant tensor and an expression.

    Computes $\mathbf{A} \cdot_k \mathbf{x}$ where $\mathbf{A}$ is a constant
    tensor, $\mathbf{x}$ is the child expression, and $k$ specifies the number
    of dimensions to contract.

    The output shape is ``A.shape[:-dims] + child.shape[dims:]``.

    See :func:`torch.tensordot` for the underlying tensor contraction semantics.
    """
    def __init__(self, value: torch.Tensor, child: ChildT, dims: int | None = None, name: str | None = None):
        if dims is None:
            dims = len(child.shape)
        assert value.shape[-dims:] == child.shape[:dims], f"Last {dims} dims of value must match first {dims} dims of child. Got value shape {value.shape} and child shape {child.shape}."
        super().__init__(ExprFlags.IS_CONST_MULTIPLICATIVE | child.flags & ExprFlags.SYMMETRIC_TO_0)
        self.value = value
        self.child = child
        self.dims = dims
        self.name = name

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self.value.shape[:-self.dims] + self.child.shape[self.dims:])
    
    @property
    def children(self) -> tuple[ChildT]:
        return (self.child,)
    
    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=","=="]="==") -> "ConstTensorDot"[ChildT]:
        return ConstTensorDot(torch.tensordot(weights, self.value, dims=self.value.dim() - self.dims), self.child, dims=self.dims)
    
    def simplify(self) -> Expr:
        """Fuse consecutive constant tensor contractions into a single operation."""
        child = self.child.simplify()
        assert child.shape == self.child.shape, "Child shape must not change during simplification."
        if isinstance(child, ConstTensorDot):
            merged_value = torch.tensordot(self.value, self.child.value, dims=self.dims)
            return ConstTensorDot(merged_value, self.child.child, dims=self.child.dims)
        return self
    
    def add_simplify(self, other: Expr) -> "ConstTensorDot"[ChildT]:
        """Merge with another ConstTensorDot sharing the same child by summing the constant tensors."""
        if isinstance(other, ConstTensorDot) and other.child is self.child:
            assert other.dims == self.dims and other.shape == self.shape, "Incompatible dimensions or shapes."
            merged_value = self.value + other.value
            return ConstTensorDot(merged_value, self.child, dims=self.dims)
        if other is self.child:
            merged_value = self.value + eye_of(self.child.shape[:self.dims])
            return ConstTensorDot(merged_value, self.child, dims=self.dims)
        return None

    def to_string(self, child_str: str) -> str:
        if self.name is not None:
            return f"tensordot(#const {self.name}, {child_str})"
        return f"tensordot(#const {self.id}, {child_str})"
    
class ConstMul[ChildT: Expr](Expr):
    r"""Element-wise multiplication between a constant tensor and an expression.

    Computes $\mathbf{A} \odot \mathbf{x}$ (Hadamard product) where $\mathbf{A}$
    is a constant tensor and $\mathbf{x}$ is the child expression.

    The shapes of the constant and child must be broadcastable, and the output
    shape follows standard PyTorch broadcasting rules.

    During backward propagation, the weight tensor is multiplied element-wise
    with the constant: $\mathbf{w}^\top (\mathbf{A} \odot \mathbf{x}) =
    (\mathbf{w} \odot \mathbf{A})^\top \mathbf{x}$.
    """
    child: ChildT
    value: torch.Tensor
    name: str | None

    def __init__(self, value: torch.Tensor, child: ChildT, name: str | None = None):
        # Verify shapes are broadcastable
        try:
            output_shape = torch.broadcast_shapes(value.shape, child.shape)
        except RuntimeError as e:
            raise ValueError(f"Shapes {value.shape} and {child.shape} are not broadcastable") from e

        super().__init__(ExprFlags.IS_CONST_MULTIPLICATIVE | child.flags & ExprFlags.SYMMETRIC_TO_0)
        self.value = value
        self.child = child
        self.name = name
        self._shape = output_shape

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def children(self) -> tuple[ChildT]:
        return (self.child,)

    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=", "=="] = "==") -> "ConstMul[ChildT]":
        # w^T (A ⊙ x) = (w ⊙ A)^T x
        new_weights = weights * self.value
        return ConstMul(new_weights, self.child)

    def simplify(self) -> Expr:
        """Fuse consecutive element-wise multiplications."""
        child = self.child.simplify()
        if isinstance(child, ConstMul):
            # (A ⊙ (B ⊙ x)) = ((A ⊙ B) ⊙ x)
            merged_value = self.value * child.value
            return ConstMul(merged_value, child.child)
        return ConstMul(self.value, child) if child is not self.child else self

    def add_simplify(self, other: Expr) -> "ConstMul[ChildT] | None":
        """Merge with another ConstMul sharing the same child."""
        if isinstance(other, ConstMul) and other.child is self.child:
            merged_value = self.value + other.value
            return ConstMul(merged_value, self.child)
        return None

    def to_string(self, child_str: str) -> str:
        if self.name is not None:
            return f"(#const {self.name} ⊙ {child_str})"
        return f"(#const {self.id} ⊙ {child_str})"


class ConstMatmul[ChildT: Expr](Expr):
    r"""Matrix multiplication between a constant matrix and an expression.

    Computes $\mathbf{A} \mathbf{x}$ where $\mathbf{A}$ is a constant 2D matrix
    and $\mathbf{x}$ is the child expression. This is a specialized form of
    :class:`ConstTensorDot` optimized for the common matrix-vector case.

    For a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and child with shape
    $(\ldots, n)$, the output has shape $(\ldots, m)$.

    During backward propagation: $\mathbf{w}^\top (\mathbf{A} \mathbf{x}) =
    (\mathbf{A}^\top \mathbf{w})^\top \mathbf{x}$.
    """
    child: ChildT
    value: torch.Tensor
    name: str | None

    def __init__(self, value: torch.Tensor, child: ChildT, name: str | None = None):
        assert value.dim() == 2, f"ConstMatmul requires a 2D matrix, got shape {value.shape}"
        assert len(child.shape) >= 1, "Child must have at least one dimension"
        assert child.shape[-1] == value.shape[-1], \
            f"Child's last dimension {child.shape[-1]} must match matrix columns {value.shape[-1]}"

        super().__init__(ExprFlags.IS_CONST_MULTIPLICATIVE | child.flags & ExprFlags.SYMMETRIC_TO_0)
        self.value = value
        self.child = child
        self.name = name

    @property
    def shape(self) -> torch.Size:
        # A: (m, n), x: (..., n) -> output: (..., m)
        return self.child.shape[:-1] + torch.Size([self.value.shape[0]])

    @property
    def children(self) -> tuple[ChildT]:
        return (self.child,)

    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=", "=="] = "==") -> "ConstMatmul[ChildT]":
        # w^T (A @ x) = (A^T @ w)^T @ x = (w @ A)^T @ x
        # weights: (..., m), A: (m, n) -> new_weights: (..., n)
        new_weights = weights @ self.value
        return ConstMatmul(new_weights, self.child)

    def simplify(self) -> Expr:
        """Fuse consecutive matrix multiplications."""
        child = self.child.simplify()
        if isinstance(child, ConstMatmul):
            # A @ (B @ x) = (A @ B) @ x
            merged_value = self.value @ child.value
            return ConstMatmul(merged_value, child.child)
        return ConstMatmul(self.value, child) if child is not self.child else self

    def add_simplify(self, other: Expr) -> "ConstMatmul[ChildT] | None":
        """Merge with another ConstMatmul sharing the same child."""
        if isinstance(other, ConstMatmul) and other.child is self.child:
            assert other.value.shape == self.value.shape, "Incompatible matrix shapes"
            merged_value = self.value + other.value
            return ConstMatmul(merged_value, self.child)
        return None

    def to_string(self, child_str: str) -> str:
        if self.name is not None:
            return f"(#const {self.name} @ {child_str})"
        return f"(#const {self.id} @ {child_str})"    

