r"""Basic Expression Classes

This module defines fundamental expression types including constants,
addition, and tensor slicing operations.
"""

from typing import Literal, TYPE_CHECKING

import torch
from sortedcontainers import SortedSet

from boundlab.expr._core import Expr, ExprFlags

if TYPE_CHECKING:
    from boundlab.expr._mul import ConstTensorDot

class ConstVal(Expr):
    r"""A leaf expression representing a constant tensor value.

    During backward propagation, this expression contributes a bias term
    computed as $\mathbf{w}^\top \mathbf{c}$, where $\mathbf{w}$ is the
    propagated weight and $\mathbf{c}$ is the constant value.
    """
    def __init__(self, value: torch.Tensor, name: str | None = None):
        super().__init__(ExprFlags.NO_DEPENTENCY | ExprFlags.IS_CONST)
        self.value = value
        self.name = name

    @property
    def shape(self) -> torch.Size:
        return self.value.shape
    @property
    def children(self) -> tuple[()]:
        return ()
    
    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=","=="]="==") -> "ConstVal":
        return ConstVal(torch.tensordot(weights, self.value, dims=self.value.dim()))
    
    def to_string(self) -> str:
        if self.name is not None:
            return "#const " + self.name
        return f"#const " + self.id
    
    def add_simplify(self, other):
        """Merge with another ConstVal by summing their values."""
        if isinstance(other, ConstVal):
            assert other.shape == self.shape, "Incompatible shapes."
            merged_value = self.value + other.value
            return ConstVal(merged_value)
        return None

class Add[*ChildT](Expr):
    r"""An expression representing the element-wise sum of child expressions.

    All children must have identical shapes. During backward propagation,
    the same weight tensor is distributed to each child term.
    """
    def __init__(self, *children: *ChildT):
        super().__init__()
        self._children = tuple(children)
        if all(ExprFlags.SYMMETRIC_TO_0 in child.flags for child in children):
            self.flags |= ExprFlags.SYMMETRIC_TO_0
        assert all(child.shape == children[0].shape for child in children), "Children must have the same shape."

    @property
    def shape(self) -> torch.Size:
        return self.children[0].shape
    
    @property
    def children(self) -> tuple[*ChildT]:
        return self._children
    
    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=","=="]="==") -> "Add":
        from boundlab.expr._mul import ConstTensorDot
        return Add(*[ConstTensorDot(weights, child) for child in self.children])

    def simplify(self):
        """Flatten nested sums and merge compatible terms."""
        simplified = AddList()
            
        for child in self.children:
            simplified_child = child.simplify()
            simplified.append(simplified_child)
        simplified.sort(key=lambda expr: expr.child.id if ExprFlags.IS_CONST_MULTIPLICATIVE in expr.flags else expr.id)
        return Add(*simplified)

    def to_string(self, *children_str: str) -> str:
        return " + ".join(children_str)

def add(*children: (Expr | torch.Tensor)) -> Add:
    """Construct an Add expression from expressions or tensors.

    Tensors are automatically wrapped in :class:`ConstVal`, and nested
    :class:`Add` expressions are flattened into a single level.
    """
    processed_children = []
    for child in children:
        if not isinstance(child, Expr):
            processed_children.append(ConstVal(torch.tensor(child)))
        elif isinstance(child, Add):
            processed_children.extend(child.children)
        else:
            processed_children.append(child)
    return Add(*processed_children)

class SubTensor[ChildT: Expr](Expr):
    r"""An expression representing indexing or slicing of a child expression.

    Supports NumPy-style advanced indexing:

    - **Integer index**: Selects a single element along the dimension,
      reducing the rank by one.
    - **Slice**: Selects a contiguous or strided range along the dimension.
    - **None**: Inserts a new axis of size 1 at the specified position.

    During backward propagation, weights are scattered back to the
    corresponding positions in the child expression's shape.
    """
    def __init__(self, child: ChildT, indices: tuple[int | slice | None, set[int]]):
        super().__init__()
        assert all(s == child.shape[0] for s in child.shape), "All elements in the input tuple must have the same shape."
        self.child = child
        self.indices = list(indices)
        if len(indices) > len(child.shape):
            raise ValueError(f"Too many indices: expected at most {len(child.shape)}, but got {len(indices)}.")
        while len(self.indices) < len(child.shape):
            self.indices.append(slice(None))

    @property
    def shape(self) -> torch.Size:
        size = []
        for i, idx in enumerate(self.indices):
            if isinstance(idx, int):
                assert self.child.shape[i] > idx >= -self.child.shape[i], f"Index {idx} out of bounds for dimension {i} with size {shape[i]}."
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(self.child.shape[i])
                size.append((stop - start + (step - 1)) // step)
            elif idx is None:
                size.append(1)
            else:
                raise ValueError(f"Invalid index type: {type(idx)}. Expected int, slice, or None.")
        return torch.Size(size)
    
    @property
    def children(self) -> tuple[ChildT]:
        return (self.child,)
    
    def backward(self, weights: torch.Tensor, mode: Literal[">=", "<=","=="]="==") -> "ConstTensorDot[ChildT]":
        from boundlab.expr._mul import ConstTensorDot
        assert weights.shape[-len(self.shape):] == self.shape, f"Weight shape {weights.shape} must match expression shape {self.shape}."

        additional_dims = len(weights.shape) - len(self.shape)
        result_shape = weights.shape[:-len(self.shape)] + self.child.shape
        result_weights = torch.zeros(result_shape, device=weights.device, dtype=weights.dtype)

        none_dims = [additional_dims + i for i, idx in enumerate(self.indices) if idx is None]
        not_none_indices = tuple(idx for idx in self.indices if idx is not None)
        result_weights[not_none_indices] = weights.squeeze(none_dims)
        return ConstTensorDot(result_weights, self.child, dims=len(self.shape))
    
    def to_string(self, child_str: str) -> str:
        return f"{child_str}{self.indices}"


class AddList:
    """A helper class for merging addition terms with a shared child expression."""
    def __init__(self, iterable=None):
        self.terms = []
        if iterable is not None:
            for item in iterable:
                self.append(item)

    def append(self, term: Expr):
        if isinstance(term, Add):
            for child in term.children:
                self.append(child)
            return
        index = 0
        merged = None
        while index < len(self.terms):
            merged = self.terms[index].add_simplify(term)
            if merged is not None:
                break
            index += 1
        if index < len(self.terms):
            self.terms.pop(index)
            self.append(merged)
        else:
            self.terms.append(term)            

class AddPriorityQueue:
    """A helper class for merging addition terms with a shared child expression, using a priority queue to optimize merging."""
    def __init__(self, iterable=None):
        self.terms = SortedSet(lambda expr: -(expr.child.id if ExprFlags.IS_CONST_MULTIPLICATIVE in expr.flags else expr.id))
        if iterable is not None:
            for item in iterable:
                self.add(item)

    def add(self, term: Expr):
        if isinstance(term, Add):
            for child in term.children:
                self.add(child)
            return
        for a in self.terms:
            merged = a.add_simplify(term)
            if merged is not None:
                self.terms.discard(a)
                self.cadd(merged)
                return
        self.terms.add(term)        

    def pop(self) -> Expr:
        return self.terms.pop()