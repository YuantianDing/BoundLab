from __future__ import annotations

r"""Utility Functions for BoundLab

This module provides helper functions used throughout the BoundLab framework.

Examples
--------
>>> from boundlab.utils import merge_name
>>> merge_name("x", "+", "y")
'(x + y)'
"""

from torch._subclasses.fake_tensor import FakeTensorMode
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, TypeAlias, TypeVar as _TypeVar, Union

if TYPE_CHECKING:
    from boundlab.expr import Expr

import torch
import copy
A = _TypeVar("A")

Triple: TypeAlias = tuple[A, A, A]

__all__ = ["Triple", "merge_name"]


def merge_name(name1, op: str, name2) -> str | None:
    """Merge two optional names into a single name for a composed operation.

    Examples
    --------
    >>> merge_name("left", "@", "right")
    '(left @ right)'
    >>> merge_name(None, "@", "right") is None
    True
    """
    name1 = name1.name if hasattr(name1, "name") else name1
    name2 = name2.name if hasattr(name2, "name") else name2
    if type(name1) is not str or type(name2) is not str:
        return None
    if name1 is not None and name2 is not None:
        return f"({name1} {op} {name2})"
    return None

def is0(input) -> bool:
    """Helper function to check if a tensor is identically zero."""
    if isinstance(input, int):
        return input == 0
    else:
        return False
def not0(input) -> bool:
    """Helper function to check if a tensor is not identically zero."""
    if isinstance(input, int):
        return input != 0 
    else:
        return True
    

def multiple_diagnonal(tensor, dims: list[tuple[int, int]]) -> tuple[torch.Tensor, list[int]]:
    """Extract multiple diagonals from a tensor iteratively.

    Each ``(dim1, dim2)`` pair is passed to :func:`torch.diagonal`, which
    removes those two dims and appends a new trailing dim holding the
    diagonal.  ``dims`` references the *original* tensor's dimensions.

    Returns:
        ``(new_tensor, dim_map)`` where ``dim_map[i]`` is the current
        position of the original dim ``i``.  Two original dims that got
        fused into the same diagonal map to the same (last) position.

    Examples:
        >>> import torch
        >>> from boundlab.utils import multiple_diagnonal
        >>> t = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        >>> out, dim_map = multiple_diagnonal(t, [(0, 1)])
        >>> out.tolist()
        [1.0, 5.0, 9.0]
        >>> dim_map
        [0, 0]
    """
    dim_map = list(range(len(tensor.shape)))
    for dim1, dim2 in dims:
        assert dim1 != dim2
        d1, d2 = dim_map[dim1], dim_map[dim2]
        tensor = torch.diagonal(tensor, dim1=d1, dim2=d2)
        last = len(tensor.shape) - 1
        new_dim_map = []
        for v in dim_map:
            if v == d1 or v == d2:
                new_dim_map.append(last)
            else:
                shift = (1 if v > d1 else 0) + (1 if v > d2 else 0)
                new_dim_map.append(v - shift)
        dim_map = new_dim_map
    return tensor, dim_map



def multiple_diag_embed(tensor, dims: Counter[int, int]) -> tuple[torch.Tensor, list]:
    """Embed specified dims as diagonals, iteratively expanding the tensor.

    Each dim in ``dims`` is turned into a ``(k, k)`` diagonal via
    :func:`torch.diag_embed`, where ``k`` is that dim's size.  ``dims``
    references the *original* tensor's dimensions.

    Returns:
        ``(new_tensor, dim_map)`` where each original dim ``i`` maps to
        either an ``int`` (its current position) or a ``(p, q)`` tuple
        of two positions when that dim was embedded as a diagonal.

    Examples:
        >>> import torch
        >>> from boundlab.utils import multiple_diag_embed
        >>> t = torch.tensor([1., 2., 3.])
        >>> out, dim_map = multiple_diag_embed(t, [0])
        >>> out.tolist()
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        >>> dim_map
        [(0, 1)]
    """
    dim_map = [[i] for i in range(len(tensor.shape))]
    for dim, count in dims.items():
        assert count >= 2
        d = dim_map[dim]
        assert len(d) == 1
        d = d[0]
        tensor = tensor.transpose(d, -1)
        last = len(tensor.shape) - 1
        # After transpose, swap positions d and last in dim_map entries.
        def _swap(v):
            return list(d if x == last else last if x == d else x for x in v)
        dim_map = [_swap(v) for v in dim_map]
        
        for _ in range(count - 1):
            tensor = torch.diag_embed(tensor, dim1=d, dim2=d + 1)
        # After diag_embed: two new dims inserted at d, d+1; existing dims >= d shift by +2.
        def _shift(v):
            return list(x + count if x >= d else x for x in v)
        dim_map = [_shift(v) for v in dim_map]
        dim_map[dim] = [d + i for i in range(count)]
    return tensor, dim_map


@dataclass
class EQCondition:
    eqclasses: set[tuple[int, ...]]

    def __post_init__(self):
        new_eqclasses = set()
        for eqclass in self.eqclasses:
            if len(eqclass) <= 1:
                continue
            if any(set(eqclass).intersection(set(c)) for c in new_eqclasses):
                raise ValueError(f"EQCondition cannot have overlapping eqclasses: {eqclass} overlaps with existing classes.")
            new_eqclasses.add(eqclass)
        self.eqclasses = new_eqclasses

    def _add_tuple(self, tup: tuple[int, ...]) -> "EQCondition":
        for eqclass in self.eqclasses:
            if any(x in eqclass for x in tup):
                new_eqclass = tuple(set(eqclass) | set(tup))
                new_eqclasses = set(c for c in self.eqclasses if c != eqclass)
                new_eqclasses.add(new_eqclass)
                return EQCondition(new_eqclasses)
        return EQCondition(self.eqclasses | {tup})
    
    def _sub_tuple(self, tup: tuple[int, ...]) -> "EQCondition":
        new_eqclasses = set()
        for eqclass in self.eqclasses:
            assert type(eqclass) == tuple
            if all(x not in tup for x in eqclass):
                new_eqclasses.add(eqclass)
            else:
                new_eqclasses.add(tuple([tup[0]] + [x for x in eqclass if x not in tup]))
        return EQCondition(new_eqclasses)

    def __add__(self, other: "EQCondition") -> "EQCondition":
        result = EQCondition({})
        for eqclass in self.eqclasses:
            result = result._add_tuple(eqclass)
        for eqclass in other.eqclasses:
            result = result._add_tuple(eqclass)
        return result
    
    def __le__(self, other: "EQCondition") -> "EQCondition":
        for eqclass in self.eqclasses:
            if not any(set(eqclass).issubset(set(eqother)) for eqother in other.eqclasses):
                return False
        return True
    
    def __ge__(self, other: "EQCondition") -> "EQCondition":
        return other.__le__(self)
    
    def __sub__(self, other: "EQCondition") -> "EQCondition":
        assert self >= other, f"Cannot subtract {other} from {self} since {self} is not a stronger condition than {other}"
        result = copy.deepcopy(self)
        for eqclass in other.eqclasses:
            result = result._sub_tuple(eqclass)
        return result
    
    def __str__(self):
        return " & ".join(" = ".join(str(i) for i in eqclass) for eqclass in self.eqclasses)
    
    def to_pairs(self):
        pairs = list()
        for eqclass in self.eqclasses:
            for i in eqclass[1:]:
                pairs.append((eqclass[0], i))
        return pairs
    
    def all_pairs(self):
        return all(len(tup) <= 2 for tup in self.eqclasses)
    
def current_fake_mode():
    mode = torch.utils._python_dispatch._get_current_dispatch_mode()
    return mode if isinstance(mode, FakeTensorMode) else None

def pairwise_diff(x: Expr, dim: int = -1) -> Expr:
    """Build ``d[..., i, j, ...] = x[..., j, ...] - x[..., i, ...]`` as a
    single :class:`EinsumOp` applied to *x*.

    Using one LinearOp (rather than two broadcasted terms combined via
    subtraction) avoids the ``SumOp`` merging two structurally similar
    :class:`ExpandOp` s that would otherwise cancel the noise contribution.
    """
    if dim < 0:
        dim += len(x.shape)
    N = x.shape[dim]
    l = x.unsqueeze(dim).expand_on(dim, N)
    r = x.unsqueeze(dim+1).expand_on(dim + 1, N)
    return r - l

def remove_diagonal(tensor: Union[torch.Tensor, Expr], dim1: int=0, dim2: int=1) -> Union[torch.Tensor, Expr]:
    """Remove the diagonal along the specified dims, returning a tensor with one fewer dimension."""
    assert dim1 == 0 and dim2 == 1
    assert tensor.shape[0] == tensor.shape[1]
    N = tensor.shape[0]
    tensor = tensor.reshape(N * N, -1)
    return tensor[1:].reshape(N-1, N+1, -1)[:, :-1].reshape(N, N-1, -1)
