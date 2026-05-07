"""Compatibility operators built on the sparse LinearOp base."""

from boundlab import utils

import torch
from boundlab.sparse.dim import Dim

from boundlab.linearop._base import DEBUG_LINEAR_OP, LinearOp, LinearOpFlags
from boundlab.linearop._debug import jacobian_from_function
from boundlab.linearop._sparse import tensor_from_edges
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor
from boundlab.sparse.tn import TN


class ScalarOp(LinearOp):
    def __init__(self, scalar: float, input_shape: torch.Size, name=None):
        self.scalar = scalar
        if scalar != 1:
            tn = TN.from_dense(torch.tensor(scalar))
        else:
            tn = TN([])
        idims = [Dim(length=s, ordering=1000.0 + float(i), name=f"i{i}") for i, s in enumerate(input_shape)]
        odims = [Dim(length=s, ordering=float(i), name=f"o{i}") for i, s in enumerate(input_shape)]
        
        ops = [
            COOSparsify.md_eye(
                Dim(length=s, ordering=i, name=f"k{i}"),
                [odims[i], idims[i]],
            )
            for i, s in enumerate(input_shape)
        ]
        
        tensor = MultiCOOTensor(tn, MultiCOOSparsify(ops))
        debug_jacobian = jacobian_from_function(input_shape, input_shape, lambda x: x * scalar) if DEBUG_LINEAR_OP else None
        
        super().__init__(
            tensor,
            idims,
            odims,
            flags=LinearOpFlags.IS_NON_NEGATIVE if scalar >= 0 else LinearOpFlags.NONE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)
        self.name = name

    def __str__(self):
        return self.name if self.name is not None else f"{self.scalar}"
    
    def __matmul__(self, other: LinearOp):
        if isinstance(other, ScalarOp):
            return ScalarOp(self.scalar * other.scalar, self.input_shape, name=utils.merge_name(self, other, f"({self}*{other})"))
        else:
            return other * self.scalar
    
    def __rmatmul__(self, other):
        if isinstance(other, ScalarOp):
            return ScalarOp(self.scalar * other.scalar, self.input_shape, name=utils.merge_name(other, self, f"({other}*{self})"))
        else:
            return other * self.scalar

class ZeroOp(LinearOp):
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size, name=None):
        input_shape = torch.Size(input_shape)
        output_shape = torch.Size(output_shape)
        input_coords = torch.empty((0, len(input_shape)), dtype=torch.long)
        output_coords = torch.empty((0, len(output_shape)), dtype=torch.long)
        tensor, input_dims, output_dims = tensor_from_edges(
            input_shape, output_shape, input_coords, output_coords, torch.empty(0)
        )
        debug_jacobian = jacobian_from_function(input_shape, output_shape, lambda x: torch.zeros(output_shape, dtype=x.dtype, device=x.device)) if DEBUG_LINEAR_OP else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)
        self.name = name

    def __str__(self):
        return self.name if self.name else "0"
    
    def __matmul__(self, other: LinearOp):
        return ZeroOp(self.input_shape, other.output_shape, name=utils.merge_name(self, other, f"({self}*{other})"))
    
    def __rmatmul__(self, other):
        return ZeroOp(other.input_shape, self.output_shape, name=utils.merge_name(other, self, f"({other}*{self})"))
    
    def __add__(self, other):
        return other
    
    def __radd__(self, other):
        return other
