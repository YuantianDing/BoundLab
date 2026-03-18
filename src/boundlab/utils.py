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