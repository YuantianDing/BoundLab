"""Compatibility operators built on the sparse LinearOp base."""

import torch
from boundlab.sparse.dim import Dim

from boundlab.linearop._base import LinearOpFlags
from boundlab.linearop._sparse import SparseLinearOp as LinearOp
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
        
        super().__init__(
            tensor,
            idims,
            odims,
            flags=LinearOpFlags.IS_NON_NEGATIVE if scalar >= 0 else LinearOpFlags.NONE,
        )
        self.name = name

    def __str__(self):
        return self.name if self.name is not None else f"{self.scalar}"


class ZeroOp(LinearOp):
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size, name=None):
        input_shape = torch.Size(input_shape)
        output_shape = torch.Size(output_shape)
        input_coords = torch.empty((0, len(input_shape)), dtype=torch.long)
        output_coords = torch.empty((0, len(output_shape)), dtype=torch.long)
        tensor, input_dims, output_dims = tensor_from_edges(
            input_shape, output_shape, input_coords, output_coords, torch.empty(0)
        )
        super().__init__(tensor, input_dims, output_dims, flags=LinearOpFlags.IS_NON_NEGATIVE)
        self.name = name

    def __str__(self):
        return self.name if self.name else "0"


class ComposedOp(LinearOp):
    def __init__(self, *ops: LinearOp):
        assert len(ops) > 0
        self.ops = list(ops)
        result = self.ops[-1]
        for op in reversed(self.ops[:-1]):
            result = op @ result
        super().__init__(result.tensor, result.input_dims, result.output_dims, result.flags)
        self._forward_fn = lambda x: _compose_forward(self.ops, x)
        self._backward_fn = lambda grad: _compose_backward(self.ops, grad)


class SumOp(LinearOp):
    def __init__(self, *ops: LinearOp):
        assert len(ops) > 0
        self.ops = []
        for op in ops:
            self.ops.extend(op.ops if isinstance(op, SumOp) else [op])
        result = self.ops[0]
        for op in self.ops[1:]:
            result = result + op
        super().__init__(result.tensor, result.input_dims, result.output_dims, result.flags)
        self._forward_fn = lambda x: sum(op.forward(x) for op in self.ops)
        self._backward_fn = lambda grad: sum(op.backward(grad) for op in self.ops)


def _compose_forward(ops, x):
    for op in reversed(ops):
        x = op.forward(x)
    return x


def _compose_backward(ops, grad):
    for op in ops:
        grad = op.backward(grad)
    return grad
