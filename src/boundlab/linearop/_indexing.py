"""Index-tensor-based LinearOp implementations.

``GetIndicesOp`` and ``SetIndicesOp`` index along a single dimension using
a tensor of indices, replacing that dimension with ``added_shape``.
"""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags, ComposedOp
from boundlab.utils import merge_name


class GetIndicesOp(LinearOp):
    """Gather elements along *dim* using an index tensor.

    output_shape = input_shape[:dim] + added_shape + input_shape[dim+1:]

    Args:
        input_shape: Shape of the source tensor.
        dim: Dimension along which to index.
        indices: Index tensor with shape ``added_shape``, values in ``[0, input_shape[dim])``.
        added_shape: Shape that replaces ``input_shape[dim]``.
    """

    def __init__(self, input_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size):
        # TODO:

    def __str__(self):
        return f"<getindices dim={self.dim} added={list(self.added_shape)}>"


class SetIndicesOp(LinearOp):
    """Scatter values along *dim* using an index tensor.

    input_shape = output_shape[:dim] + added_shape + output_shape[dim+1:]

    Args:
        output_shape: Shape of the output tensor (zeros template).
        dim: Dimension along which to scatter.
        indices: Index tensor with shape ``added_shape``, values in ``[0, output_shape[dim])``.
        added_shape: Shape that replaces ``output_shape[dim]`` in the input.
    """

    def __init__(self, output_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size):
        self.dim = dim
        self.indices = indices
        self.added_shape = added_shape
        assert indices.shape == added_shape, \
            f"indices.shape={indices.shape} != added_shape={added_shape}"
        input_shape = torch.Size(
            list(output_shape[:dim]) + list(added_shape) + list(output_shape[dim + 1:])
        )
        # TODO
    def __str__(self):
        return f"<setindices dim={self.dim} added={list(self.added_shape)}>"

