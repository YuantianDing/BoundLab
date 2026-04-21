"""Permutation and transposition LinearOp implementations."""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags
from boundlab.utils import merge_name


class PermuteOp(LinearOp):
    """Permute dimensions of the input tensor."""

    def __init__(self, input_shape: torch.Size, dims: tuple[int, ...]):
        self.dims = list(dims)
        self.inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            self.inv_dims[d] = i
        output_shape = torch.Size(input_shape[d] for d in dims)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.permute(*self.dims)

    def backward(self, grad):
        return grad.permute(*self.inv_dims)

    def vforward(self, x):
        n = len(self.dims)
        batch_ndim = x.dim() - n
        perm = self.dims + [n + i for i in range(batch_ndim)]
        return x.permute(*perm)

    def vbackward(self, grad):
        n = len(self.inv_dims)
        batch_ndim = grad.dim() - n
        perm = list(range(batch_ndim)) + [batch_ndim + d for d in self.inv_dims]
        return grad.permute(*perm)

    def __matmul__(self, other):
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.input_shape == other.output_shape
            new_output_dims = [other.output_dims[self.dims[i]] for i in range(len(other.output_dims))]
            result = EinsumOp(other.tensor, other.input_dims, new_output_dims, name=merge_name(self, "@", other))
            assert result.input_shape == other.input_shape, f"PermuteOp.__matmul__: input_shape {result.input_shape} != {other.input_shape}"
            assert result.output_shape == self.output_shape, f"PermuteOp.__matmul__: output_shape {result.output_shape} != {self.output_shape}"
            return result
        return NotImplemented

    def __rmatmul__(self, other):
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.output_shape == other.input_shape
            new_input_dims = [other.input_dims[self.inv_dims[i]] for i in range(len(other.input_dims))]
            result = EinsumOp(other.tensor, new_input_dims, other.output_dims, name=merge_name(other, "@", self))
            assert result.input_shape == self.input_shape, f"PermuteOp.__rmatmul__: input_shape {result.input_shape} != {self.input_shape}"
            assert result.output_shape == other.output_shape, f"PermuteOp.__rmatmul__: output_shape {result.output_shape} != {other.output_shape}"
            return result
        return super().__rmatmul__(other)

    def __str__(self):
        return f"<permute {self.dims}>"


class TransposeOp(PermuteOp):
    """Swap two dimensions of the input tensor — special case of PermuteOp."""

    def __init__(self, input_shape: torch.Size, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1
        dims = list(range(len(input_shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        super().__init__(input_shape, tuple(dims))

    def __str__(self):
        return f"<transpose {self.dim0} {self.dim1}>"
