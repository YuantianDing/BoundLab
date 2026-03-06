r"""Linearization Strategies for Nonlinear Activations

This module provides linearizers that compute linear relaxations of
nonlinear activation functions for zonotope-based bound propagation.
"""

from dataclasses import dataclass
import typing

import torch


@dataclass
class ZonoBounds:
    """Linear relaxation bounds for a nonlinear activation.

    Attributes:
        generator: Generator tensor for the epsilon terms.
        bias: Bias term.
        weights: Coefficients for the linear approximation.
    """
    generator: torch.Tensor
    bias: torch.Tensor
    weights: torch.Tensor


ZonoLinearizer = typing.Callable[[torch.Tensor, torch.Tensor], ZonoBounds]
"""Type alias for a linearizer function mapping (upper, lower) bounds to ZonoBounds."""

ZonoLinearizers = typing.Dict[str, ZonoLinearizer]
"""Type alias for a dictionary mapping activation names to linearizers."""


def relu_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> ZonoBounds:
    r"""Compute a linear relaxation of the ReLU activation.

    Given element-wise bounds $[\ell, u]$ on the input, this function
    returns a linear relaxation of $\mathrm{ReLU}(x) = \max(0, x)$.

    Args:
        ub: Element-wise upper bounds on the input.
        lb: Element-wise lower bounds on the input.

    Returns:
        A :class:`ZonoBounds` containing the linear relaxation parameters.
    """
    pass
    

DEFAULT_LINEARIZER = {
    "relu": relu_linearizer,
}