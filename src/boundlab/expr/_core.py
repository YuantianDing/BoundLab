r"""Core Expression Classes

This module defines the base expression class and flags used throughout
the expression framework.
"""

from copy import copy
import itertools
import sys
from typing import Callable, Literal, Union

from numpy import indices
import torch
import enum

from boundlab.linearop import ScalarOp
from boundlab.linearop._base import LinearOp


class ExprFlags(enum.Flag):
    """Flags indicating expression properties for optimization."""
    NONE = 0
    SYMMETRIC_TO_0 = enum.auto()
    """The feasible region is symmetric about zero."""
    PRINT_FUSE = enum.auto()
    """Expression should be fused with parent when printing."""
    IS_CONST = enum.auto()
    """Expression represents a constant value."""
    IS_AFFINE = enum.auto()
    """Expression represents an affine transformation of its inputs."""

_EXPR_ID_COUNTER = itertools.count()

class Expr:
    """Abstract base class for all symbolic expressions in BoundLab.

    Each expression represents a node in a directed acyclic graph (DAG) of
    computations. Expressions are immutable after construction, and each
    instance is assigned a unique time-ordered UUID to enable deterministic
    topological ordering during bound propagation.

    Subclasses must implement :attr:`shape`, :attr:`children`, and
    :meth:`backward` to define the expression's semantics.
    """

    id: int
    """Unique identifier for the expression, used for topological sorting."""
    flags: ExprFlags
    """Flags indicating expression properties for optimization."""

    def __init__(self, flags: ExprFlags = ExprFlags.NONE):
        self.id = next(_EXPR_ID_COUNTER)
        self.flags = flags
    
    def with_children(self, *new_children: "Expr") -> "Expr":
        """Return a new expression with the same type and flags but new children."""
        raise NotImplementedError(f"The :code:`with_children` method is not implemented for {self.__class__.__name__}.")

    @property
    def shape(self) -> torch.Size:
        """The shape of the output(s) produced by this expression."""
        raise NotImplementedError(f"The :code:`shape` property is not implemented for {self.__class__.__name__}.")

    @property
    def children(self) -> tuple["Expr", ...]:
        """The child expressions that serve as inputs to this expression."""
        raise NotImplementedError(f"The :code:`children` property is not implemented for {self.__class__.__name__}.")

    def backward(self, weights: LinearOp, direction: Literal[">=", "<=", "=="] = "==") \
            -> tuple[torch.Tensor, list[LinearOp]] | None:
        r"""Perform backward-mode bound propagation through this expression.

        Given an accumulated weight ``weights`` (usually a 
        :class:`~boundlab.linearop.EinsumOp`) from the output back to this
        node, backward propagation derives child weights and a bias:

        .. math::

            \mathbf{w}^\top f(x_1, \ldots, x_n)
            \;\square\; b + \sum_i \mathbf{w}_i^\top x_i

        Args:
            weights: A :class:`~boundlab.linearop.EinsumOp` accumulated
                weight from the root expression to this node.
            direction: Bound direction — ``">="`` (lower), ``"<="`` (upper),
                or ``"=="`` (exact).

        Returns:
            A tuple ``(bias, child_weights)`` where ``bias`` is a
            :class:`torch.Tensor` or ``0``, and ``child_weights`` is a list
            of :class:`~boundlab.linearop.EinsumOp` (one per child).
            Returns ``None`` if this expression cannot contribute to the bound
            in the given direction.
        """
        raise NotImplementedError(f"The :code:`backward` method is not implemented for {self.__class__.__name__}.")

    def to_string(self, *children_str: str) -> str:
        """Return string representation with child strings substituted."""
        return f"{self.__class__.__name__}({', '.join(children_str)})"

    def __str__(self):
        return "bl.Expr {\n" + expr_pretty_print(self, indent=4) + "\n}"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, flags={self.flags})"

    # ------------------------------------------------------------------
    # Arithmetic operators — all produce Linear with EinsumOp ops
    # ------------------------------------------------------------------

    def __add__(self, other):
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, torch.Tensor):
            if other.shape != self.shape:
                other = other.expand(self.shape)
            other = AffineSum(const=other)
        if isinstance(other, Expr):
            assert self.shape == other.shape
            return AffineSum((ScalarOp(1.0, self.shape), self), (ScalarOp(1.0, other.shape), other))
        return NotImplemented

    def __radd__(self, other):
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, torch.Tensor):
            if other.shape != self.shape:
                other = other.expand(self.shape)
            other = AffineSum(const=other)
        if isinstance(other, Expr):
            return AffineSum((ScalarOp(1.0, other.shape), other), (ScalarOp(1.0, self.shape), self))
        return NotImplemented

    def __neg__(self):
        from boundlab.expr._affine import AffineSum, ConstVal
        from boundlab.linearop import EinsumOp
        return AffineSum((ScalarOp(-1.0, self.shape), self))

    def __sub__(self, other):
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value
        if isinstance(other, Expr):
            return self + (-other)
        if isinstance(other, torch.Tensor):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other):
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value
        if isinstance(other, Expr):
            return other + (-self)
        if isinstance(other, torch.Tensor):
            return other + (-self)
        return NotImplemented

    def __mul__(self, other):
        """Element-wise multiplication (no broadcast)."""
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value
            
        from boundlab.linearop import EinsumOp, ScalarOp
        if isinstance(other, (int, float)):
            return AffineSum((ScalarOp(float(other), self.shape), self))
        if isinstance(other, torch.Tensor):
            if other.dim() == 0:
                return AffineSum((ScalarOp(float(other.item()), self.shape), self))
            return AffineSum((EinsumOp.from_hardmard(other, len(self.shape)), self))
        return NotImplemented

    def __rmul__(self, other):
        """Element-wise multiplication (no broadcast)."""
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value

        from boundlab.linearop import EinsumOp, ScalarOp
        if isinstance(other, (int, float)):
            return AffineSum((ScalarOp(float(other), self.shape), self))
        if isinstance(other, torch.Tensor):
            if other.dim() == 0:
                return AffineSum((ScalarOp(float(other.item()), self.shape), self))
            return AffineSum((EinsumOp.from_hardmard(other, len(self.shape)), self))
        return NotImplemented

    def __truediv__(self, other):
        """Division by a scalar or tensor."""
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value
    

        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        if isinstance(other, torch.Tensor):
            if len(self.shape) == 0 and other.dim() > 0:
                return self.expand(*list(other.shape)) * (1.0 / other)
            if other.dim() > 0 and tuple(self.shape) != tuple(other.shape):
                other = other.expand(*list(self.shape))
            return self * (1.0 / other)
        return NotImplemented

    def __matmul__(self, other):
        """Matrix multiply: self @ other. Supports batched matmul."""
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value

        from boundlab.linearop import EinsumOp
        if isinstance(other, torch.Tensor) and len(other.shape) >= 2:
            assert self.shape[-1] == other.shape[-2], \
                f"Inner dims must match: {self.shape} @ {other.shape}"

            if len(self.shape) == 1:
                # Vector-matrix: (K,) @ (*batch_t, K, S2) → (*batch_t, S2)
                K = self.shape[0]
                batch_t = other.shape[:-2]
                S2 = other.shape[-1]
                nb = len(batch_t)
                # tensor = other, shape (*batch_t, K, S2)
                # input_dims: K is at nb, output_dims: (*batch, S2) = [0..nb-1, nb+1]
                input_dims = [nb]  # K
                output_dims = list(range(nb)) + [nb + 1]  # (*batch_t, S2)
                return AffineSum((EinsumOp(other, input_dims, output_dims), self))

            # self: (*batch_s, S, D), other: (*batch_t, D, S2)
            batch_s = self.shape[:-2]
            batch_t = other.shape[:-2]
            S, D, S2 = self.shape[-2], self.shape[-1], other.shape[-1]
            batch = torch.broadcast_shapes(batch_s, batch_t) if (batch_s and batch_t) else (batch_s or batch_t)
            nb = len(batch)

            # Build tensor of shape (*batch, S, D, S2)
            other_b = other.expand(*batch, D, S2)
            tensor = other_b.unsqueeze(nb).expand(*batch, S, D, S2)

            # input_dims covers self's shape within the tensor
            # self may have fewer batch dims than the tensor (broadcasting)
            ns = len(self.shape)
            input_dims = list(range(nb + 2))[-ns:]  # last ns of (*batch, S, D)
            output_dims = list(range(nb + 1)) + [nb + 2]  # (*batch, S, S2)
            return AffineSum((EinsumOp(tensor, input_dims, output_dims), self))
        return NotImplemented

    def __rmatmul__(self, other):
        """Matrix multiply: other @ self. Supports batched matmul.

        Handles both matrix-vector and matrix-matrix cases.
        """
        from boundlab.expr._affine import AffineSum, ConstVal
        if isinstance(other, ConstVal):
            other = other.value
            
        from boundlab.linearop import EinsumOp
        if isinstance(other, torch.Tensor) and len(other.shape) >= 2:
            M, K = other.shape[-2], other.shape[-1]
            batch_t = other.shape[:-2]

            if len(self.shape) == 1:
                # Matrix-vector: (*batch_t, M, K) @ (K,) → (*batch_t, M)
                assert self.shape[0] == K, f"Inner dims must match: {other.shape} @ {self.shape}"
                nb = len(batch_t)
                # EinsumOp: tensor(*batch, M, K), input=[nb+1], output=[0..nb, nb]
                # But we also need batch dims as batch_dims of EinsumOp
                input_dims = [nb + 1]  # K
                output_dims = list(range(nb + 1))  # (*batch, M)
                return AffineSum((EinsumOp(other, input_dims, output_dims), self))

            # Matrix-matrix: (*batch_t, M, K) @ (*batch_s, K, N) → (*batch, M, N)
            assert self.shape[-2] == K, f"Inner dims must match: {other.shape} @ {self.shape}"
            N = self.shape[-1]
            batch_s = self.shape[:-2]
            batch = torch.broadcast_shapes(batch_t, batch_s) if (batch_t and batch_s) else (batch_t or batch_s)
            nb = len(batch)

            # Build tensor of shape (*batch, M, K, N)
            other_b = other.expand(*batch, M, K)
            tensor = other_b.unsqueeze(-1).expand(*batch, M, K, N)

            # input_dims: self's shape (*batch_s, K, N) maps to tensor dims
            ns = len(self.shape)
            # K is at nb+1, N is at nb+2; batch dims are the last len(batch_s) of 0..nb-1
            input_dims = list(range(nb))[-len(batch_s):] + [nb + 1, nb + 2] if batch_s else [nb + 1, nb + 2]
            output_dims = list(range(nb + 1)) + [nb + 2]  # (*batch, M, N)
            return AffineSum((EinsumOp(tensor, input_dims, output_dims), self))
        return NotImplemented
    

    # ------------------------------------------------------------------
    # Shape / indexing operators — produce Linear with generic LinearOp
    # ------------------------------------------------------------------

    def _apply_op(self, op):
        """Wrap a LinearOp as an AffineSum expression."""
        from boundlab.expr._affine import AffineSum
        return AffineSum((op, self))

    def reshape(self, *shape) -> "Expr":
        from boundlab.linearop import ReshapeOp
        return self._apply_op(ReshapeOp(self.shape, shape))

    def permute(self, *dims) -> "Expr":
        from boundlab.linearop import PermuteOp
        return self._apply_op(PermuteOp(self.shape, dims))

    def transpose(self, dim0, dim1) -> "Expr":
        from boundlab.linearop import TransposeOp
        return self._apply_op(TransposeOp(self.shape, dim0, dim1))
    
    @property
    def T(self) -> "Expr":
        """Convenience for transpose of the last two dimensions."""
        if len(self.shape) < 2:
            raise ValueError(f"Cannot transpose last two dimensions of shape {self.shape} with less than 2 dims.")
        return self.transpose(len(self.shape) - 2, len(self.shape) - 1)

    def flatten(self, start_dim=0, end_dim=-1) -> "Expr":
        from boundlab.linearop import FlattenOp
        return self._apply_op(FlattenOp(self.shape, start_dim, end_dim))

    def unflatten(self, dim, sizes) -> "Expr":
        from boundlab.linearop import UnflattenOp
        return self._apply_op(UnflattenOp(self.shape, dim, sizes))

    def squeeze(self, dim=None) -> "Expr":
        from boundlab.linearop import SqueezeOp
        return self._apply_op(SqueezeOp(self.shape, dim))

    def unsqueeze(self, dim) -> "Expr":
        from boundlab.linearop import UnsqueezeOp
        return self._apply_op(UnsqueezeOp(self.shape, dim))

    def mean(self, dim=None, keepdim=False) -> "Expr":
        from boundlab.linearop import ReduceMeanOp
        if dim is None:
            dim = tuple(range(len(self.shape)))
        elif isinstance(dim, int):
            dim = (dim,)
        return self._apply_op(ReduceMeanOp(self.shape, dim, keepdim=keepdim))

    def narrow(self, dim, start, length) -> "Expr":
        from boundlab.linearop import NarrowOp
        return self._apply_op(NarrowOp(self.shape, dim, start, length))

    def expand(self, *sizes) -> "Expr":
        from boundlab.linearop import ExpandOp
        return self._apply_op(ExpandOp(self.shape, sizes))

    def repeat(self, *sizes) -> "Expr":
        from boundlab.linearop import RepeatOp
        return self._apply_op(RepeatOp(self.shape, sizes))

    def tile(self, *sizes) -> "Expr":
        from boundlab.linearop import TileOp
        return self._apply_op(TileOp(self.shape, sizes))

    def flip(self, dims) -> "Expr":
        from boundlab.linearop import FlipOp
        return self._apply_op(FlipOp(self.shape, dims))

    def roll(self, shifts, dims) -> "Expr":
        from boundlab.linearop import RollOp
        return self._apply_op(RollOp(self.shape, shifts, dims))

    def diag(self, diagonal=0) -> "Expr":
        from boundlab.linearop import DiagOp
        return self._apply_op(DiagOp(self.shape, diagonal))

    def __getitem__(self, indices) -> "Expr":
        from boundlab.linearop import GetItemOp, GetIndicesOp
        if not isinstance(indices, tuple):
            indices = (indices,)
        if all(isinstance(idx, slice) or isinstance(idx, int) for idx in indices):
            return self._apply_op(GetItemOp(self.shape, indices))
        elif all(isinstance(idx, torch.Tensor) and idx.dtype == torch.bool for idx in indices):
            return self._apply_op(GetIndicesOp(self.shape, indices))
        raise ValueError("Invalid indices for item selection")

    def zeros_set(self, output_shape) -> Callable[[tuple], "Expr"]:
        from boundlab.linearop import SetSliceOp, SetIndicesOp
        def zeros_set(indices):
            if not isinstance(indices, tuple):
                indices = (indices,)
            if all(isinstance(idx, slice) or isinstance(idx, int) for idx in indices):
                return self._apply_op(SetSliceOp(indices, self.shape, output_shape))
            elif all(isinstance(idx, torch.Tensor) and idx.dtype == torch.bool for idx in indices):
                return self._apply_op(SetIndicesOp(indices, self.shape, output_shape))
            else:
                raise ValueError("Invalid indices for zero setting")
        return zeros_set
    
    def scatter(self, indices, output_shape) -> "Expr":
        from boundlab.linearop import ScatterOp
        return self._apply_op(ScatterOp(indices, self.shape, output_shape))
    
    def gather(self, indices) -> "Expr":
        from boundlab.linearop import GatherOp
        return self._apply_op(GatherOp(indices, self.shape))

    # ------------------------------------------------------------------
    # Bound computation helpers
    # ------------------------------------------------------------------

    def ub(self) -> torch.Tensor:
        """Compute an upper bound for this expression."""
        from boundlab import prop
        return prop.ub(self)

    def lb(self) -> torch.Tensor:
        """Compute a lower bound for this expression."""
        from boundlab import prop
        return prop.lb(self)

    def ublb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both an upper bound and a lower bound for this expression."""
        from boundlab import prop
        return prop.ublb(self)

    def center(self) -> torch.Tensor:
        """Compute the center of the bounds for this expression."""
        from boundlab import prop
        return prop.center(self)

    def bound_width(self) -> torch.Tensor:
        """Compute the width of the bounds for this expression."""
        from boundlab import prop
        return prop.bound_width(self)
    
    def get_const(self):
        """Return the concrete tensor if *self* is a pure constant expression, else None.

        Works for :class:`~boundlab.expr.ConstVal` and any :class:`~boundlab.expr.AffineSum`
        that has no symbolic children.
        """
        return None
    
    def simplify_ops_(self):
        """Recursively compute simplified ops for affine expressions."""
        pass

    def is_symmetric_to_0(self) -> bool:
        """Return True if this expression is symmetric about zero, else False."""
        return bool(self.flags & ExprFlags.SYMMETRIC_TO_0)

    def symmetric_decompose(self) -> tuple["Expr | 0", "Expr | 0"]:
        """Decompose this expression into a constant part and a zero-constant part.

        If the expression is symmetric about zero, the constant part is zero.
        If the expression is a pure constant, the zero-constant part is zero.

        Returns:
            A tuple ``(non_const_part, const_part)`` where exactly one of the two is zero.
        """
        raise NotImplementedError(f"The :code:`symmetric_decompose` method is not implemented for {self.__class__.__name__}.")


def expr_pretty_print(expr: Expr, indent: int = 0) -> str:
    """Pretty print an expression in SSA form."""
    visited = list()

    def dfs(e: Expr):
        if e in visited:
            return
        visited.append(e)
        for child in e.children:
            dfs(child)
    dfs(expr)

    visited.reverse()

    ref_count = {e: 0 for e in visited}
    for e in visited:
        for child in e.children:
            ref_count[child] += 1

    def can_fuse(e: Expr) -> bool:
        return (ExprFlags.PRINT_FUSE in e.flags) and ref_count[e] <= 1

    expr_to_str = {}

    def get_expr_str(e: Expr) -> str:
        if e in expr_to_str:
            return expr_to_str[e]
        children_strs = []
        for child in e.children:
            if can_fuse(child):
                children_strs.append(get_expr_str(child))
            else:
                children_strs.append(f'%{visited.index(child)}')
        result = e.to_string(*children_strs)
        expr_to_str[e] = result
        return result

    output = []
    for i, e in enumerate(visited):
        if can_fuse(e):
            continue
        get_expr_str(e)
        output.append((" " * indent) + f"%{i} = {expr_to_str[e]}")
    return "\n".join(output)
