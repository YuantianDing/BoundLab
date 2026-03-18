r"""Utility Functions for BoundLab

This module provides helper functions used throughout the BoundLab framework.
"""

import string
from typing import Callable, Literal
import torch


def eye_of(shape: torch.Size) -> torch.Tensor:
    """Create an identity tensor of the given shape.

    The output is a tensor of shape ``shape + shape`` where the last two
    dimensions form an identity matrix. This is used for propagating bounds
    through addition operations.

    Args:
        shape: The shape of the leading dimensions.
    Returns:
        An identity tensor of shape ``shape + shape``.
    """
    if len(shape) == 1:
        return torch.eye(shape[0])
    else:
        # TODO: Implement for higher dimensions if needed
        raise NotImplementedError("eye_of is only implemented for 1D shapes for now.")
    
def merge_name(name1, op: str, name2) -> str | None:
    """Merge two optional names into a single name for a composed operation."""
    name1 = name1.name if hasattr(name1, "name") else name1
    name2 = name2.name if hasattr(name2, "name") else name2
    if type(name1) is not str or type(name2) is not str:
        return None
    if name1 is not None and name2 is not None:
        return f"({name1} {op} {name2})"
    return None