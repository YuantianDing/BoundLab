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
        from boundlab.expr._affine import AffineSum
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
        from boundlab.expr._affine import AffineSum
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
        from boundlab.expr._affine import AffineSum
        from boundlab.linearop import EinsumOp
        return AffineSum((ScalarOp(-1.0, self.shape), self))

    def __sub__(self, other):
        if isinstance(other, Expr):
            return self + (-other)
        if isinstance(other, torch.Tensor):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Expr):
            return other + (-self)
        if isinstance(other, torch.Tensor):
            return other + (-self)
        return NotImplemented

    def __mul__(self, other):
        """Element-wise multiplication (no broadcast)."""
        from boundlab.expr._affine import AffineSum
        from boundlab.linearop import EinsumOp, ScalarOp
        if isinstance(other, (int, float)):
            return AffineSum((ScalarOp(float(other), self.shape), self))
        if isinstance(other, torch.Tensor):
            return AffineSum((EinsumOp.from_hardmard(other, len(self.shape)), self))
        return NotImplemented

    def __rmul__(self, other):
        """Element-wise multiplication (no broadcast)."""
        from boundlab.expr._affine import AffineSum
        from boundlab.linearop import EinsumOp, ScalarOp
        if isinstance(other, (int, float)):
            return AffineSum((ScalarOp(float(other), self.shape), self))
        if isinstance(other, torch.Tensor):
            return AffineSum((EinsumOp.from_hardmard(other, len(self.shape)), self))
        return NotImplemented

    def __truediv__(self, other):
        """Division by a scalar or tensor."""
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        if isinstance(other, torch.Tensor):
            return self * (1.0 / other)
        return NotImplemented

    def __matmul__(self, other):
        """Matrix multiply: self @ other."""
        from boundlab.expr._affine import AffineSum
        from boundlab.linearop import EinsumOp
        if isinstance(other, torch.Tensor) and len(other.shape) == 2:
            assert self.shape[-1] == other.shape[0], f"Inner dimension of self {self.shape} must match first dimension of other {other.shape} for matmul."
            tensor = other[tuple([None] * len(self.shape[:-1]) + [slice(None), slice(None)])].expand(*self.shape[:-1], *other.shape)
            input_dims = list(range(len(self.shape)))
            output_dims = input_dims[:-1] + [tensor.dim() - 1]
            return AffineSum((EinsumOp(tensor, input_dims, output_dims), self))
        return NotImplemented

    def __rmatmul__(self, other):
        """Matrix multiply: other @ self.

        Handles both:
        - Tensor(m, k) @ Expr(k,)   → Expr(m,)      (matrix-vector)
        - Tensor(m, k) @ Expr(k, n) → Expr(m, n)     (matrix-matrix)
        """
        from boundlab.expr._affine import AffineSum
        from boundlab.linearop import EinsumOp
        if isinstance(other, torch.Tensor) and len(other.shape) == 2:
            m, k = other.shape
            if len(self.shape) == 1:
                # Matrix-vector: (m, k) @ (k,) → (m,)
                assert self.shape[0] == k, f"Inner dims must match: {other.shape} @ {self.shape}"
                # EinsumOp: tensor(m,k), input_dims=[1], output_dims=[0]
                return AffineSum((EinsumOp(other, input_dims=[1], output_dims=[0]), self))
            elif len(self.shape) == 2:
                # Matrix-matrix: (m, k) @ (k, n) → (m, n)
                k2, n = self.shape
                assert k == k2, f"Inner dims must match: {other.shape} @ {self.shape}"
                tensor3d = other.unsqueeze(2).expand(m, k, n)
                return AffineSum((EinsumOp(tensor3d, input_dims=[1, 2], output_dims=[0, 2]), self))
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
