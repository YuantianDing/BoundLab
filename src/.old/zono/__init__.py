r"""Zonotope-Based Abstract Interpretation for Neural Networks

This module provides zonotope transformers for computing over-approximations
of neural network outputs under bounded input perturbations.
"""

from typing import Callable, TypeVar

import torch
from torch import nn

from boundlab.expr import Expr, ConstVal, ConstMul, ConstMatmul, LInfEps, add
from boundlab.prop import ub, lb
from .linearizer import ZonoLinearizers, DEFAULT_LINEARIZER


# Type for layer handlers
T = TypeVar('T', bound=nn.Module)
LayerHandler = Callable[[T, Expr, ZonoLinearizers], Expr]

# Registry for layer handlers
_LAYER_HANDLERS: dict[type, LayerHandler] = {}


def register_handler(layer_type: type[T]):
    """Decorator to register a handler for a specific layer type.

    Args:
        layer_type: The nn.Module subclass this handler processes.

    Returns:
        A decorator that registers the handler function.
    """
    def decorator(handler: LayerHandler[T]) -> LayerHandler[T]:
        _LAYER_HANDLERS[layer_type] = handler
        return handler
    return decorator


def transform_layer(layer: nn.Module, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Transform an expression through a single layer.

    Args:
        layer: The layer to transform through.
        x: The input expression.
        linearizers: Dictionary of linearization functions for activations.

    Returns:
        The transformed expression.

    Raises:
        NotImplementedError: If no handler is registered for the layer type.
    """
    # Look up handler for this layer type
    for layer_type, handler in _LAYER_HANDLERS.items():
        if isinstance(layer, layer_type):
            return handler(layer, x, linearizers)

    raise NotImplementedError(
        f"No handler registered for layer type {type(layer).__name__}. "
        f"Supported types: {list(_LAYER_HANDLERS.keys())}"
    )


# =============================================================================
# Layer Handlers
# =============================================================================

@register_handler(nn.Linear)
def _handle_linear(layer: nn.Linear, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.Linear: y = Wx + b."""
    weight = layer.weight.detach()  # Shape: (out_features, in_features)
    result = ConstMatmul(weight, x)

    if layer.bias is not None:
        bias = layer.bias.detach()
        result = add(result, ConstVal(bias))

    return result


@register_handler(nn.ReLU)
def _handle_relu(layer: nn.ReLU, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.ReLU using the registered linearizer."""
    return _apply_linearizer("relu", x, linearizers)


@register_handler(nn.LeakyReLU)
def _handle_leaky_relu(layer: nn.LeakyReLU, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.LeakyReLU using the registered linearizer."""
    # Use leaky_relu linearizer if available, otherwise fall back
    name = f"leaky_relu_{layer.negative_slope}"
    if name in linearizers:
        return _apply_linearizer(name, x, linearizers)
    elif "leaky_relu" in linearizers:
        return _apply_linearizer("leaky_relu", x, linearizers)
    else:
        raise NotImplementedError(
            f"No linearizer found for LeakyReLU. "
            f"Register 'leaky_relu' or '{name}' in linearizers."
        )


@register_handler(nn.Sigmoid)
def _handle_sigmoid(layer: nn.Sigmoid, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.Sigmoid using the registered linearizer."""
    return _apply_linearizer("sigmoid", x, linearizers)


@register_handler(nn.Tanh)
def _handle_tanh(layer: nn.Tanh, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.Tanh using the registered linearizer."""
    return _apply_linearizer("tanh", x, linearizers)


@register_handler(nn.Sequential)
def _handle_sequential(layer: nn.Sequential, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.Sequential by processing layers in order."""
    result = x
    for sublayer in layer:
        result = transform_layer(sublayer, result, linearizers)
    return result


@register_handler(nn.Flatten)
def _handle_flatten(layer: nn.Flatten, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.Flatten - for now, assume input is already flat."""
    # TODO: Implement proper flatten handling with SubTensor/reshape
    # For simple cases where input is 1D, this is a no-op
    return x


@register_handler(nn.Identity)
def _handle_identity(layer: nn.Identity, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.Identity - pass through unchanged."""
    return x


@register_handler(nn.Dropout)
def _handle_dropout(layer: nn.Dropout, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.Dropout - treated as identity during verification."""
    return x


@register_handler(nn.BatchNorm1d)
def _handle_batchnorm1d(layer: nn.BatchNorm1d, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Handle nn.BatchNorm1d in eval mode as an affine transformation."""
    if layer.training:
        raise ValueError("BatchNorm must be in eval mode for verification")

    # BatchNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
    # In eval mode, this is an affine transformation: y = scale * x + bias
    mean = layer.running_mean.detach()
    var = layer.running_var.detach()
    eps = layer.eps
    gamma = layer.weight.detach() if layer.affine else torch.ones_like(mean)
    beta = layer.bias.detach() if layer.affine else torch.zeros_like(mean)

    scale = gamma / torch.sqrt(var + eps)
    bias_val = beta - mean * scale

    result = ConstMul(scale, x)
    result = add(result, ConstVal(bias_val))
    return result


# =============================================================================
# Helper Functions
# =============================================================================

def _apply_linearizer(name: str, x: Expr, linearizers: ZonoLinearizers) -> Expr:
    """Apply a linearizer to transform an expression through a nonlinear activation.

    Args:
        name: The name of the linearizer to use.
        x: The input expression.
        linearizers: Dictionary of linearization functions.

    Returns:
        The transformed expression: weights * x + bias + generator * fresh_eps
    """
    if name not in linearizers:
        raise NotImplementedError(
            f"No linearizer found for '{name}'. "
            f"Available linearizers: {list(linearizers.keys())}"
        )

    linearizer = linearizers[name]

    # Compute bounds on the input expression
    x_ub = ub(x)
    x_lb = lb(x)

    # Get linearization parameters
    bounds = linearizer(x_ub, x_lb)

    # Construct output: y = weights * x + bias + generator * eps_new
    # Start with the linear part
    result = ConstMul(bounds.weights, x)

    # Add the bias
    result = add(result, ConstVal(bounds.bias))

    # Add fresh noise symbol scaled by generator (only where generator != 0)
    if (bounds.generator != 0).any():
        fresh_eps = LInfEps(*bounds.generator.shape)
        eps_term = ConstMul(bounds.generator, fresh_eps)
        result = add(result, eps_term)

    return result


# =============================================================================
# Main Operator Function
# =============================================================================

def operator(net: nn.Module, linearizers: ZonoLinearizers = DEFAULT_LINEARIZER):
    r"""Construct a zonotope transformer for a neural network.

    Given a network module and linearization strategies for nonlinear
    activations, this function returns an operator that propagates
    zonotope abstractions through the network.

    The transformation proceeds layer by layer, applying exact affine
    transformations for linear layers and introducing fresh noise symbols
    via linearization for nonlinear activations.

    **Supported layers:**

    - ``nn.Linear``: Exact affine transformation
    - ``nn.ReLU``: Triangle relaxation (requires 'relu' linearizer)

    To add support for additional layers, use :func:`register_handler`.

    Args:
        net: The neural network module to transform.
        linearizers: A dictionary mapping activation function names to linearizer
            functions. Defaults to :data:`DEFAULT_LINEARIZER`.

    Returns:
        A callable that maps input expressions to output expressions
        representing the zonotope abstraction of the network's output.

    Example:
        >>> net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
        >>> transform = operator(net)
        >>> x = LInfEps((2,))  # Input perturbation
        >>> y = transform(x)   # Output zonotope expression
    """
    def operation(x: Expr) -> Expr:
        return transform_layer(net, x, linearizers)

    return operation

