import torch

from boundlab.expr._core import Expr
from boundlab.linearop import LinearOp
from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp

from . import ZonoBounds, _register_linearizer


@_register_linearizer("relu")
def relu_linearizer(expr: Expr) -> ZonoBounds:
    """Triangle relaxation of ReLU for zonotope abstract interpretation.

    For each neuron with input bounds [lb, ub]:
    - Dead   (ub <= 0): output is 0, no contribution.
    - Active (lb >= 0): output equals input exactly.
    - Crossing (lb < 0 < ub): triangle relaxation with
        slope  = ub / (ub - lb),
        bias   = -ub * lb / (2 * (ub - lb)),
        error  = -ub * lb / (2 * (ub - lb)).
    """

    lb = expr.lb()
    ub = expr.ub()
    output_shape = ub.shape
    dead   = ub <= 0
    active = lb >= 0
    cross  = ~dead & ~active

    slope = torch.where(active, torch.ones_like(ub), torch.zeros_like(ub))
    slope = torch.where(cross, ub / (ub - lb), slope)

    error = torch.zeros_like(ub)
    cross_val = -ub * lb / (2 * (ub - lb))
    bias  = torch.where(cross, cross_val, torch.zeros_like(ub))
    nonzero_idx = torch.nonzero(cross, as_tuple=True)
    length = nonzero_idx[0].shape[0]
    cross_coeffs = cross_val[nonzero_idx]
    indices_op = SetIndicesOp(nonzero_idx, torch.Size((length,)), output_shape) 
    hardmard_op = EinsumOp.from_hardmard(cross_coeffs, 1)
    hardmard_op.flags |= LinearOpFlags.IS_NON_NEGATIVE
    return ZonoBounds(bias=bias, error_coeffs=indices_op @ hardmard_op, input_weights=[slope])
