"""Einsum-based sparse LinearOp."""

import torch

from boundlab.linearop._base import DEBUG_LINEAR_OP, LinearOp, LinearOpFlags
from boundlab.linearop._debug import jacobian_from_function
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor
from boundlab.sparse.dim import Dim
from boundlab.sparse.tn import Dense, TN


class EinsumOp(LinearOp):
    r"""Linear operator defined by an Einstein summation with a fixed tensor."""

    def __init__(self, tensor: torch.Tensor, input_dims: list[int], output_dims: list[int], name=None):
        self.coeff = tensor
        self.einsum_input_axes = list(input_dims)
        self.einsum_output_axes = list(output_dims)
        self.dot_dims = [i for i in input_dims if i not in output_dims]
        self.mul_dims = [i for i in output_dims if i in input_dims]
        self.batch_dims = [i for i in output_dims if i not in input_dims]
        input_shape = torch.Size(self.coeff.shape[i] for i in input_dims)
        output_shape = torch.Size(self.coeff.shape[i] for i in output_dims)
        lin_input_dims = [Dim(length=s, ordering=1000.0 + i, name=f"i{i}") for i, s in enumerate(input_shape)]
        lin_output_dims = [Dim(length=s, ordering=float(i), name=f"o{i}") for i, s in enumerate(output_shape)]
        tensor_dims = [Dim(length=s, ordering=float(i), name=f"t{i}") for i, s in enumerate(tensor.shape)]
        ops = []
        
        for i, dim in enumerate(tensor_dims):
            dims = []
            if i in input_dims:
                idx = input_dims.index(i)
                dims.append(lin_input_dims[idx])
            if i in output_dims:
                idx = output_dims.index(i)
                dims.append(lin_output_dims[idx])
            assert dims
            ops.append(COOSparsify.md_eye(dim, dims))
        dense = Dense(self.coeff, tensor_dims)
        sparse_tensor = MultiCOOTensor(TN.from_dense(dense), MultiCOOSparsify(ops))
        debug_jacobian = jacobian_from_function(
            input_shape,
            output_shape,
            lambda x: torch.einsum(
                self.coeff,
                list(range(self.coeff.ndim)),
                x,
                self.einsum_input_axes,
                self.einsum_output_axes,
            ),
        ) if DEBUG_LINEAR_OP else None

        super().__init__(
            sparse_tensor,
            lin_input_dims,
            lin_output_dims,
            flags=LinearOpFlags.NONE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            print(self.tensor.to_dense(), self.debug_jacobian)
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __str__(self):
        if self.name:
            return self.name
        return f"<einsum {list(self.coeff.shape)}: {self.einsum_input_axes} -> {self.einsum_output_axes}>"

    @staticmethod
    def from_hardmard(tensor: torch.Tensor, n_input_dims: int = None, name=None) -> "EinsumOp":
        if n_input_dims is None:
            n_input_dims = tensor.dim()
        output_dims = list(range(tensor.dim()))
        input_dims = output_dims[-n_input_dims:]
        return EinsumOp(tensor, input_dims, output_dims, name=name)

    @staticmethod
    def from_full(tensor: torch.Tensor, input_dim: int, name=None) -> "EinsumOp":
        output_dims = list(range(tensor.dim() - input_dim))
        input_dims = list(range(tensor.dim() - input_dim, tensor.dim()))
        return EinsumOp(tensor, input_dims, output_dims, name=name)
