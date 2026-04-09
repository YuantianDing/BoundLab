import torch

from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._einsum import EinsumOp
from boundlab.linearop._indices import SetIndicesOp

from . import ZonoBounds, _register_linearizer


@_register_linearizer("relu")
def relu_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> ZonoBounds:
    """Triangle relaxation of ReLU for zonotope abstract interpretation.

    For each neuron with input bounds [lb, ub]:

    - Dead (ub <= 0): output is 0, no contribution.
    - Active (lb >= 0): output equals input exactly.
    - Crossing (lb < 0 < ub): triangle relaxation with
      slope = ub / (ub - lb),
      bias = -ub * lb / (2 * (ub - lb)),
      error = -ub * lb / (2 * (ub - lb)).
    """

    output_shape = ub.shape
    lam = (torch.relu(ub) - torch.relu(lb)) / (ub - lb + 1e-30)
    mu = 0.5 * (torch.relu(ub) - lam * ub)
    # nonzero_idx = torch.nonzero(cross, as_tuple=True)
    # length = nonzero_idx[0].shape[0]
    # cross_coeffs = cross_val[nonzero_idx]
    # indices_op = SetIndicesOp(nonzero_idx, torch.Size((length,)), output_shape) 
    # hardmard_op = EinsumOp.from_hardmard(cross_coeffs, 1)
    # hardmard_op.flags |= LinearOpFlags.IS_NON_NEGATIVE
    hardmard_op = EinsumOp.from_hardmard(mu, len(ub.shape))
    return ZonoBounds(bias=mu, error_coeffs=hardmard_op, input_weights=[lam])
