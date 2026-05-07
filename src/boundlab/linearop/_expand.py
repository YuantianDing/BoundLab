"""Expand LinearOp implementation."""

import torch

from boundlab.linearop._base import DEBUG_LinearOp, LinearOp, LinearOpFlags
from boundlab.linearop._compat import ScalarOp
from boundlab.linearop._sparse import make_input_dims, make_output_dims
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import TN


def _expand_tensor(input_shape: torch.Size, output_shape: torch.Size):
    n_new = len(output_shape) - len(input_shape)
    padded_input_shape = torch.Size([1] * n_new + list(input_shape))
    input_dims = make_input_dims(input_shape)
    output_dims = make_output_dims(output_shape)
    ops = []

    for axis, output_dim in enumerate(output_dims):
        input_dim = None if axis < n_new else input_dims[axis - n_new]
        input_size = padded_input_shape[axis]
        output_size = output_shape[axis]
        assert input_size == 1 or input_size == output_size, (
            f"ExpandOp: dim {axis} has input size {input_size} and output size {output_size}."
        )

        if input_dim is not None and input_size == output_size:
            inner = Dim(int(output_size), 500.0 + axis, f"k{axis}")
            ops.append(COOSparsify.md_eye(inner, [output_dim, input_dim]))
            continue

        edge = Dim(int(output_size), 500.0 + axis, f"k{axis}")
        columns = [output_dim]
        data = [torch.arange(output_size, dtype=torch.long)]
        if input_dim is not None:
            columns.append(input_dim)
            data.append(torch.zeros(output_size, dtype=torch.long))
        ops.append(COOSparsify(edge, TorchTable(columns=columns, data=data, length=int(output_size))))

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
        debug_jacobian = tensor.to_dense().expand(output_dims + input_dims) if DEBUG_LinearOp else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LinearOp:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)
        self.name = f"<expand {list(input_shape)} -> {list(output_shape)}>"
