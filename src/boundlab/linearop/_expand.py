"""Expand LinearOp implementation."""

import torch

from boundlab.linearop._base import DEBUG_LINEAR_OP, LinearOp, LinearOpFlags
from boundlab.linearop._compat import ScalarOp
from boundlab.linearop._debug import jacobian_from_function
from boundlab.linearop._sparse import make_input_dims, make_output_dims
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import TN


def _expand_tensor(input_shape: torch.Size, output_shape: torch.Size):
    n_new = len(output_shape) - len(input_shape)
    input_dims = make_input_dims(input_shape)
    output_dims = make_output_dims(output_shape)
    ops = []

    for axis, output_dim in enumerate(output_dims):
        input_dim = None if axis < n_new else input_dims[axis - n_new]
        if input_dim and input_dim.length > 1:
            assert input_dim.length == output_dim.length, f"Cannot expand dimension {axis} of size {input_dim.length} to size {output_dim.length}"
            ops.append(COOSparsify.md_eye(input_dim.clone(suffix=output_dim.name), [input_dim, output_dim]))
        elif input_dim and input_dim.length == 1:
            ops.append(COOSparsify.md_eye(input_dim.clone(suffix="I"), [input_dim]))
            ops.append(COOSparsify.md_eye(output_dim.clone(suffix="I"), [output_dim]))
        else:
            ops.append(COOSparsify.md_eye(output_dim.clone(suffix="I"), [output_dim]))

    tensor = MultiCOOTensor(TN(factors=[]), MultiCOOSparsify(ops))
    return MultiCOOTensorSum([tensor]), input_dims, output_dims


class ExpandOp(LinearOp):
    def __new__(cls, input_shape: torch.Size, output_shape: torch.Size):
        input_shape = torch.Size(input_shape)
        output_shape = torch.Size(output_shape)
        n_new = len(output_shape) - len(input_shape)
        assert n_new >= 0, (
            f"ExpandOp: output cannot have fewer dims than input ({len(output_shape)} < {len(input_shape)})"
        )
        if input_shape == output_shape:
            return ScalarOp(1.0, input_shape, name=f"<expand {list(input_shape)} -> {list(output_shape)}>")
        return super().__new__(cls)

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        input_shape = torch.Size(input_shape)
        output_shape = torch.Size(output_shape)
        self._padded_input_shape = torch.Size([1] * (len(output_shape) - len(input_shape)) + list(input_shape))
        tensor, input_dims, output_dims = _expand_tensor(input_shape, output_shape)
        debug_jacobian = jacobian_from_function(input_shape, output_shape, lambda x: x.expand(output_shape)) if DEBUG_LINEAR_OP else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)
        self.name = f"<expand {list(input_shape)} -> {list(output_shape)}>"
