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

    The relaxation uses the standard "triangle" method:

    - **Inactive** ($u \leq 0$): $y = 0$
    - **Active** ($\ell \geq 0$): $y = x$
    - **Crossing** ($\ell < 0 < u$): Linear approximation centered between
      the upper bound line $y = \lambda (x - \ell)$ where $\lambda = u/(u-\ell)$
      and the lower bound $y = 0$.

    For the crossing case, the output is:

    $$y = \lambda x + \mu \pm \delta$$

    where:
    - $\lambda = u / (u - \ell)$ (slope)
    - $\mu = -u\ell / (2(u - \ell))$ (bias, centers the approximation)
    - $\delta = -u\ell / (2(u - \ell))$ (generator, captures error)

    Args:
        ub: Element-wise upper bounds on the input.
        lb: Element-wise lower bounds on the input.

    Returns:
        A :class:`ZonoBounds` containing the linear relaxation parameters.
    """
    # Initialize output tensors
    weights = torch.zeros_like(ub)
    bias = torch.zeros_like(ub)
    generator = torch.zeros_like(ub)

    # Case 1: Inactive region (ub <= 0)
    # ReLU(x) = 0, so weights = 0, bias = 0, generator = 0
    inactive = ub <= 0
    # Already initialized to zero, nothing to do

    # Case 2: Active region (lb >= 0)
    # ReLU(x) = x, so weights = 1, bias = 0, generator = 0
    active = lb >= 0
    weights[active] = 1.0

    # Case 3: Crossing region (lb < 0 < ub)
    # Use triangle relaxation centered between upper bound and y = 0
    crossing = (lb < 0) & (ub > 0)

    if crossing.any():
        ub_cross = ub[crossing]
        lb_cross = lb[crossing]

        # Slope of upper bound line: λ = ub / (ub - lb)
        slope = ub_cross / (ub_cross - lb_cross)

        # Bias and generator to center between upper bound and y = 0
        # At x = 0: upper bound = -λ * lb = λ * |lb|
        # Center = λ * |lb| / 2, generator = λ * |lb| / 2
        half_error = -ub_cross * lb_cross / (2 * (ub_cross - lb_cross))

        weights[crossing] = slope
        bias[crossing] = half_error
        generator[crossing] = half_error

    return ZonoBounds(generator=generator, bias=bias, weights=weights)

# TODO: Implement additional linearizers for other activation functions (e.g., sigmoid, tanh, x^2, etc.).
    

DEFAULT_LINEARIZER = {
    "relu": relu_linearizer,
}