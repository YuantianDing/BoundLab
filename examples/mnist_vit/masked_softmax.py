"""Masked softmax: custom ONNX op and abstract interpretation handlers.

Provides a ``MaskedSoftmax`` custom ONNX op with zonotope and differential
handlers that follow the DeepT decomposition internally (pairwise diff →
exp linearizer → mask concrete coefficients → sum → reciprocal).

.. note::

   This module is transitional.  Once the Heaviside-based decomposition
   (exp → heaviside_pruning → sum → reciprocal) is verified equivalent,
   this file and the ``MaskedSoftmax`` op are deleted.
"""
from __future__ import annotations

import torch
from torch import Tensor

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab import utils
from boundlab.expr._affine import ConstVal
from boundlab.expr._core import Expr
from boundlab.expr._var import LpEpsilon
from boundlab.interp import _onnx_broadcast
from boundlab.zono.bilinear import bilinear_elementwise
from boundlab.zono.exp import exp_linearizer
from boundlab.zono.reciprocal import reciprocal_linearizer


# ---------------------------------------------------------------------------
# Register bilinear elementwise Mul (needed for attn_w @ V in diff path)
# ---------------------------------------------------------------------------

def _mul_handler(X, Y):
    X, Y = _onnx_broadcast(X, Y)
    if isinstance(X, Expr) and isinstance(Y, Expr):
        return bilinear_elementwise(X, Y)
    return X * Y

zono.interpret["Mul"] = _mul_handler


# ---------------------------------------------------------------------------
# Custom ONNX op: MaskedSoftmax
# ---------------------------------------------------------------------------

def masked_softmax_op(scores: Tensor, col_mask: Tensor) -> Tensor:
    """Custom ONNX op for masked softmax.

    At concrete runtime returns zeros (use ``for_verification=False`` for MC).
    During ONNX export creates a ``boundlab::MaskedSoftmax`` node that the
    zonotope and differential interpreters handle.

    Args:
        scores: Raw attention scores, shape ``(h, n, n)``.
        col_mask: Column mask, shape ``(n,)``.  1 for kept, 0 for pruned.
    """
    return torch.onnx.ops.symbolic(
        "boundlab::MaskedSoftmax",
        (scores, col_mask),
        dtype=scores.dtype,
        shape=scores.shape,
        version=1,
    )


# ---------------------------------------------------------------------------
# Zonotope handler
# ---------------------------------------------------------------------------

def masked_softmax_zono_handler(x, col_mask) -> Expr:
    """Zonotope masked softmax via DeepT decomposition.

    Pairwise diff → exp linearizer on concrete bounds → multiply col_mask
    into concrete coefficients → sum → reciprocal.  No bilinear Expr × Expr
    products.

    Math::

        masked_softmax(s)_k = col_mask[k] / Σ_j col_mask[j] * exp(s_j - s_k)
    """
    if isinstance(x, torch.Tensor):
        x = ConstVal(x)
    if not isinstance(x, Expr):
        return NotImplemented

    if isinstance(col_mask, Expr):
        col_mask = prop.center(col_mask)

    dim = len(x.shape) - 1
    diff = -utils.pairwise_diff(x, dim)
    ub, lb = diff.ublb()

    expbounds = exp_linearizer(ub, lb)
    weights = expbounds.input_weights[0]
    bias = expbounds.bias
    error = expbounds.error_coeffs.tensor

    finite_mask = (torch.isfinite(weights) & torch.isfinite(error)
                   & torch.isfinite(bias) & (lb < 30) & (ub < 30))
    weights = torch.where(finite_mask, weights, 0)
    bias = torch.where(finite_mask, bias, 0)
    error = torch.where(finite_mask, error, 0)

    # Mask: multiply col_mask into concrete coefficients before sum
    weights = weights * col_mask
    bias = bias * col_mask
    error = error * col_mask

    # Sum over j (last dim)
    sum_exp = ((weights * diff).sum(dim=-1)
               + error.sum(dim=-1) * LpEpsilon(diff.shape[:-2], reason="masked_softmax_exp")
               + bias.sum(dim=-1))
    finite_mask = finite_mask.all(dim=-1)

    # Tighten sum bounds
    sum_exp_ub, sum_exp_lb = sum_exp.ublb()
    sum_exp_ub = torch.minimum(sum_exp_ub, (torch.exp(ub) * col_mask).sum(dim=-1))
    sum_exp_lb = torch.maximum(sum_exp_lb, (torch.exp(lb) * col_mask).sum(dim=-1))

    bounds = reciprocal_linearizer(sum_exp_ub, sum_exp_lb)
    w = bounds.input_weights[0]
    mu = bounds.bias
    beta = bounds.error_coeffs.tensor

    result = finite_mask * (w * sum_exp + mu
                            + beta * LpEpsilon(sum_exp.shape, reason="masked_softmax_recip"))

    target_shape = list(result.shape)
    col_mask_nd = col_mask.view(*([1] * (len(target_shape) - 1)), -1).expand(target_shape).contiguous()
    result = result * col_mask_nd

    return result


zono.interpret["MaskedSoftmax"] = masked_softmax_zono_handler


# ---------------------------------------------------------------------------
# Differential handler
# ---------------------------------------------------------------------------

def diff_masked_softmax_handler(x, col_mask):
    """Differential masked softmax handler.

    Follows ``diff_softmax_handler`` pattern: pairwise diff → diff exp →
    mask (const multiply, preserves correlations) → sum → diff reciprocal.

    Falls back to ``masked_softmax_zono_handler`` for plain Expr input.
    """
    from boundlab.diff.expr import DiffExpr2, DiffExpr3
    from boundlab.diff.zono3 import interpret as _diff_interpret

    if isinstance(col_mask, (DiffExpr2, DiffExpr3)):
        col_mask_x = prop.center(col_mask.x) if isinstance(col_mask.x, Expr) else col_mask.x
        col_mask_y = prop.center(col_mask.y) if isinstance(col_mask.y, Expr) else col_mask.y
    elif isinstance(col_mask, torch.Tensor):
        col_mask_x = col_mask_y = col_mask
    elif isinstance(col_mask, Expr):
        col_mask_x = col_mask_y = prop.center(col_mask)
    else:
        return NotImplemented

    delta_mask = col_mask_x - col_mask_y

    if isinstance(x, Expr) and not isinstance(x, (DiffExpr2, DiffExpr3)):
        return masked_softmax_zono_handler(x, col_mask_x)

    if isinstance(x, DiffExpr2):
        x = DiffExpr3(x.x, x.y, x.x - x.y)
    if not isinstance(x, DiffExpr3):
        return NotImplemented

    exp_handler = _diff_interpret["Exp"]
    reciprocal_handler = _diff_interpret["Reciprocal"]

    dim = len(x.shape) - 1
    n = x.shape[dim]

    # 1. Pairwise diff
    x_i = x.unsqueeze(dim + 1)
    x_j = x.unsqueeze(dim)
    broadcast_shape = list(x.shape)
    broadcast_shape.insert(dim + 1, n)
    x_i_exp = x_i.expand(*broadcast_shape)
    x_j_exp = x_j.expand(*broadcast_shape)
    x_shifted = x_j_exp - x_i_exp

    # 2. Exp
    exp_shifted = exp_handler(x_shifted)

    # 3. Mask on j-axis
    j_axis = dim + 1
    mask_shape = [1] * len(exp_shifted.shape)
    mask_shape[j_axis] = n
    col_mask_x_j = col_mask_x.view(mask_shape).expand(exp_shifted.shape).contiguous()
    col_mask_y_j = col_mask_y.view(mask_shape).expand(exp_shifted.shape).contiguous()
    delta_mask_j = col_mask_x_j - col_mask_y_j

    exp_masked = DiffExpr3(
        exp_shifted.x * col_mask_x_j,
        exp_shifted.y * col_mask_y_j,
        exp_shifted.x * delta_mask_j + exp_shifted.diff * col_mask_y_j,
    )

    # 4. Sum over j
    sum_exp = DiffExpr3(
        exp_masked.x.sum(dim=j_axis, keepdim=False),
        exp_masked.y.sum(dim=j_axis, keepdim=False),
        exp_masked.diff.sum(dim=j_axis, keepdim=False),
    )

    # 5. Reciprocal
    result = reciprocal_handler(sum_exp)

    # 6. Mask on output
    k_shape = [1] * len(result.shape)
    k_shape[-1] = n
    col_mask_x_k = col_mask_x.view(k_shape).expand(result.shape).contiguous()
    col_mask_y_k = col_mask_y.view(k_shape).expand(result.shape).contiguous()
    delta_mask_k = col_mask_x_k - col_mask_y_k

    result_final = DiffExpr3(
        result.x * col_mask_x_k,
        result.y * col_mask_y_k,
        result.x * delta_mask_k + result.diff * col_mask_y_k,
    )

    return result_final


from boundlab.diff.zono3 import interpret as diff_interpret
diff_interpret["MaskedSoftmax"] = diff_masked_softmax_handler