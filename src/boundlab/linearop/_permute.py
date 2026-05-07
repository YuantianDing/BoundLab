"""Permutation and transposition LinearOp implementations."""

import torch

from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._sparse import SparseLinearOp as LinearOp
from boundlab.linearop._sparse import make_input_dims, make_output_dims
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.tn import TN


class PermuteOp(LinearOp):
    def __init__(self, input_shape: torch.Size, dims: tuple[int, ...]):
        input_shape = torch.Size(input_shape)
        ndim = len(input_shape)
        dims = tuple(dim + ndim if dim < 0 else dim for dim in dims)
        assert len(dims) == ndim, f"Expected {ndim} permutation dims, got {len(dims)}."
        assert set(dims) == set(range(ndim)), f"Invalid permutation dims {dims} for shape {input_shape}."

        self.dims = list(dims)
        self.inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            self.inv_dims[d] = i
        output_shape = torch.Size(input_shape[d] for d in dims)
        input_dims = make_input_dims(input_shape)
        output_dims = make_output_dims(output_shape)
        ops = []
        for output_axis, input_axis in enumerate(dims):
            inner = Dim(
                length=int(input_shape[input_axis]),
                ordering=500.0 + float(output_axis),
                name=f"k{output_axis}",
            )
            ops.append(
                COOSparsify.md_eye(
                    inner,
                    [output_dims[output_axis], input_dims[input_axis]],
                )
            )

        tensor = MultiCOOTensor(TN(factors=[]), MultiCOOSparsify(ops))
        super().__init__(
            MultiCOOTensorSum([tensor]),
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
        )

    def __str__(self):
        return f"<permute {self.dims}>"


class TransposeOp(PermuteOp):
    def __init__(self, input_shape: torch.Size, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1
        dims = list(range(len(input_shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        super().__init__(input_shape, tuple(dims))

    def __str__(self):
        return f"<transpose {self.dim0} {self.dim1}>"
