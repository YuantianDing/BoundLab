r"""Linear Operations for Expressions

This module provides VJP-based linear operation wrappers that enable
automatic backward propagation through arbitrary linear transformations.
"""

import inspect
from typing import Callable, Literal

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from boundlab.expr._base import Add
from boundlab.expr._core import Expr, ExprFlags


class LinearOp:
    r"""A wrapper for linear functions with automatic VJP computation.

    This class wraps a linear function and computes its vector-Jacobian
    product (VJP) using ``torch.func.vjp``. The VJP enables efficient
    backward propagation of weights through the linear transformation.

    The wrapped function must satisfy linearity: ``f(0) = 0``.
    """

    original: Callable[[torch.Tensor], torch.Tensor]
    """The original linear function."""
    input_shape: "torch.Size"
    """Expected input tensor shape."""
    output_shape: "torch.Size"
    """Computed output tensor shape."""
    transposed: Callable[["torch.Tensor"], tuple["torch.Tensor", ...]]
    """The VJP function for backward propagation."""

    def __init__(self, original: Callable[[torch.Tensor], torch.Tensor], input_shape: torch.Size, name=None):
        """Initialize a LinearOp wrapper.

        Args:
            original: A linear function mapping tensors to tensors.
            input_shape: The expected shape of input tensors.
            name: Optional name for display purposes.

        Raises:
            AssertionError: If the function is not linear (f(0) != 0).
        """
        self.original = original
        self.input_shape = input_shape
        self.name = name
        with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
            self.output_shape = self.original(torch.empty(self.input_shape)).shape
        bias, vjp = torch.func.vjp(self.original, torch._efficientzerotensor(self.input_shape))
        assert bias._is_zerotensor(), "Original function must be linear (zero input should yield zero output)."
        self.transposed = vjp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear operation to a batched input.

        Args:
            x: Input tensor with a leading batch dimension.

        Returns:
            Output tensor with the operation applied to each batch element.
        """
        return torch.vmap(self.original)(x)

    def backward(self, weights: torch.Tensor) -> torch.Tensor:
        """Propagate weights backward through the linear operation.

        Args:
            weights: Weight tensor with a leading batch dimension.

        Returns:
            Propagated weights via the VJP.
        """
        # VJP returns a tuple of tangents, extract the first (and only) element
        return torch.vmap(self.transposed)(weights)[0]
    
    def __str__(self):
        if self.original.__name__ == "<lambda>":
            file = inspect.getsourcefile(self.original)
            line_no = inspect.getsourcelines(self.original)[1]
            return f"<lambda {file}:{line_no}>"
        return self.name if self.name is not None else self.original.__name__


class LinearOpSeq(Expr):
    r"""An expression representing a sequence of composed linear operations.

    This expression applies a sequence of :class:`LinearOp` transformations
    to a child expression. During backward propagation, weights are
    propagated through each operation in sequence via their VJPs.

    Attributes:
        child: The input expression.
        ops: List of LinearOp operations to apply (in forward order).
    """

    def __init__(self, ops: list[LinearOp | Callable[[torch.Tensor], torch.Tensor]], x: Expr):
        """Initialize a LinearOpSeq expression.

        Args:
            ops: Sequence of linear operations or callables to apply.
            x: The input expression to transform.

        Raises:
            AssertionError: If x is already a LinearOpSeq (use merging instead).
        """
        super().__init__()
        assert not isinstance(x, LinearOpSeq), "Nested LinearOpSeq is not allowed. Please merge the operations into a single sequence (using ``linear_seq``)."
        self.child = x
        
        self.ops = []
        shape = x.shape
        for op in reversed(ops):
            if isinstance(op, LinearOp):
                linear_op = op
                assert linear_op.input_shape == shape, f"Shape mismatch: expected {linear_op.input_shape}, got {shape}."
                self.ops.append(linear_op)
                shape = linear_op.output_shape
            else:
                linear_op = LinearOp(op, shape)
                self.ops.append(linear_op)
                shape = linear_op.output_shape        
        self.ops.reverse()
        
    @property
    def shape(self) -> torch.Size:
        return self.ops[0].output_shape
    
    @property
    def children(self) -> tuple[Expr]:
        return (self.child,)

    def with_children(self, new_child: Expr) -> "LinearOpSeq":
        """Return a new LinearOpSeq with the same operations but a new child."""
        return linear_op(*self.ops)(new_child)

    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=", "=="] = "==") -> tuple[0, torch.Tensor] | None:
        for op in self.ops:
            weights = op.backward(weights)
        return (0, weights)
    
    def to_string(self, *children_str):
        return " ∘ ".join([str(op) for op in self.ops]) + f"({children_str[0]})"

    def composed(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def composed_function(x: torch.Tensor) -> torch.Tensor:
            for op in self.ops:
                x = op.original(x)
            return x
        return composed_function
    
    def contract(self) -> "LinearOpSeq":
        """Attempt to contract the linear operations into a single operation, if possible."""
        jacobian = torch.func.jacrev(self.composed())(torch._efficientzerotensor(self.child.shape))
        return LinearOpSeq([TensorDotLinearOp(jacobian, dims=self.child.shape.numel())], self.child)
    

def linear_op(*ops: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[Expr], LinearOpSeq]:
    """Create a function that applies a sequence of linear operations to an expression."""
    def apply_linear_seq(x: Expr) -> LinearOpSeq:
        if isinstance(x, LinearOpSeq):
            return LinearOpSeq(ops + [op for op in x.ops], x.child)
        return LinearOpSeq(list(ops), x)
    return apply_linear_seq


class TensorDotLinearOp(LinearOp):
    r"""A LinearOp implementing tensor dot product with a fixed tensor.

    This is a specialized LinearOp that computes ``torch.tensordot(tensor, x, dims)``.

    Attributes:
        tensor: The fixed tensor for the dot product.
        dims: Number of dimensions to contract.
    """

    def __init__(self, tensor: torch.Tensor, dims: int, name=None):
        """Initialize a TensorDotLinearOp.

        Args:
            tensor: The fixed tensor for the dot product.
            dims: Number of trailing dimensions to contract.
            name: Optional name for display purposes.
        """
        self.tensor = tensor
        self.dims = dims
        super().__init__(
            original=lambda x: torch.tensordot(self.tensor, x, dims=self.dims),
            input_shape=torch.Size(self.tensor.shape[-self.dims:]),
            name=f"<tensordot {name}>" if name is not None else f"<tensordot {self.tensor.shape[-self.dims:]}>"
        )

def contract_linear_ops(expr: Expr) -> Expr:
    """Contract a LinearOpSeq into a single TensorDotLinearOp.

    Args:
        expr: A LinearOpSeq expression to contract.

    Returns:
        A new LinearOpSeq with a single contracted operation.

    Raises:
        TypeError: If expr is not a LinearOpSeq.
    """
    expr = expr.with_children(*[contract_linear_ops(child) for child in expr.children])
    if isinstance(expr, LinearOpSeq):
        return expr.contract()
    if isinstance(expr, Add):
        other_exprs = []
        expr.children_map = {}
        for child in expr.children:
            if isinstance(child, LinearOpSeq):
                assert len(child.ops) == 1, "Expected a single operation after contraction."
                assert isinstance(child.ops[0], TensorDotLinearOp), "Expected a TensorDotLinearOp after contraction."
                if child.child in expr.children_map:
                    expr.children_map[child.child] = child.ops[0].tensor
                else:
                    expr.children_map[child.child] += child.ops[0].tensor
            else:
                other_exprs.append(child)
        
        new_children = other_exprs
        new_children.extend([LinearOpSeq([TensorDotLinearOp(v, dims=k.shape.numel())], k) for k, v in expr.children_map.items()])
        return torch.add(*new_children)
    return expr