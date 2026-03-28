r"""Utility Functions for BoundLab

This module provides helper functions used throughout the BoundLab framework.

Examples
--------
>>> from boundlab.utils import merge_name
>>> merge_name("x", "+", "y")
'(x + y)'
"""

import string
from typing import Callable, Literal, TypeAlias, TypeVar

A = TypeVar("A")

Triple: TypeAlias = tuple[A, A, A]
import torch

    
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
