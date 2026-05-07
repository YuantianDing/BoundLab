"""Reshape LinearOp implementations."""

import torch

from boundlab.linearop._base import DEBUG_LINEAR_OP, LinearOp, LinearOpFlags
from boundlab.linearop._debug import jacobian_from_function
from boundlab.linearop._sparse import make_input_dims, make_output_dims, unravel
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import TN


def _meta_output_shape(fn, input_shape: torch.Size) -> torch.Size:
    return fn(torch.empty(input_shape, device="meta")).shape


def _prod(shape: torch.Size) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _coords(shape: torch.Size) -> torch.Tensor:
    flat = torch.arange(_prod(shape), dtype=torch.long)
    return unravel(flat, shape)


def _reshape_unit_axes(input_shape: torch.Size, output_shape: torch.Size) -> tuple[list[int], list[int]]:
    squeezed_input_axes = [axis for axis, size in enumerate(input_shape) if size == 1]
    unsqueezed_output_axes = [axis for axis, size in enumerate(output_shape) if size == 1]
    return squeezed_input_axes, unsqueezed_output_axes


def _groups(input_shape: torch.Size, output_shape: torch.Size):
    i = j = 0
    while i < len(input_shape) or j < len(output_shape):
        input_start, output_start = i, j
        input_numel = output_numel = 1
        if i < len(input_shape):
            input_numel *= int(input_shape[i])
            i += 1
        if j < len(output_shape):
            output_numel *= int(output_shape[j])
            j += 1
        while input_numel != output_numel:
            if (input_numel < output_numel and i < len(input_shape)) or j == len(output_shape):
                input_numel *= int(input_shape[i])
                i += 1
            else:
                output_numel *= int(output_shape[j])
                j += 1
        yield list(range(input_start, i)), list(range(output_start, j))


def _reshape_tensor(input_shape: torch.Size, output_shape: torch.Size):
    input_dims = make_input_dims(input_shape)
    output_dims = make_output_dims(output_shape)
    squeezed_input_axes, unsqueezed_output_axes = _reshape_unit_axes(input_shape, output_shape)
    input_core_axes = [axis for axis in range(len(input_shape)) if axis not in squeezed_input_axes]
    output_core_axes = [axis for axis in range(len(output_shape)) if axis not in unsqueezed_output_axes]
    input_core_shape = torch.Size(input_shape[axis] for axis in input_core_axes)
    output_core_shape = torch.Size(output_shape[axis] for axis in output_core_axes)
    ops = []
    for group_idx, (input_group, output_group) in enumerate(_groups(input_core_shape, output_core_shape)):
        input_axes = [input_core_axes[axis] for axis in input_group]
        output_axes = [output_core_axes[axis] for axis in output_group]
        in_dims = [input_dims[axis] for axis in input_axes]
        out_dims = [output_dims[axis] for axis in output_axes]
        if len(input_axes) == len(output_axes) == 1:
            inner = Dim(int(input_shape[input_axes[0]]), 500.0 + group_idx, f"k{group_idx}")
            ops.append(COOSparsify.md_eye(inner, out_dims + in_dims))
            continue

        in_shape = torch.Size(input_shape[axis] for axis in input_axes)
        out_shape = torch.Size(output_shape[axis] for axis in output_axes)
        length = _prod(in_shape)
        edge = Dim(length, 500.0 + group_idx, f"k{group_idx}")
        input_coords = _coords(in_shape)
        output_coords = _coords(out_shape)
        data = [
            *(output_coords[:, idx].contiguous() for idx in range(len(out_dims))),
            *(input_coords[:, idx].contiguous() for idx in range(len(in_dims))),
        ]
        ops.append(
            COOSparsify(
                edge,
                TorchTable(
                    columns=out_dims + in_dims, data=data, length=length,
                    is_unique=True,
                ),
            )
        )
    unit_group_idx = len(ops)
    for axis in unsqueezed_output_axes:
        unit_group_idx += 1
        assert output_dims[axis].length == 1
        ops.append(
            COOSparsify.md_eye(output_dims[axis], [output_dims[axis]])
        )
    for axis in squeezed_input_axes:
        edge = Dim(1, 500.0 + unit_group_idx, f"k{unit_group_idx}")
        unit_group_idx += 1
        ops.append(
            COOSparsify.md_eye(input_dims[axis], [input_dims[axis]])
        )
    # if len(ops) == 0:
    #     edge = Dim(1, 500.0, "k0")
    #     ops.append(COOSparsify(edge, TorchTable(columns=[], data=[], length=1)))
    tensor = MultiCOOTensor(TN(factors=[]), MultiCOOSparsify(ops))
    return MultiCOOTensorSum([tensor]), input_dims, output_dims


def _reshape_zero_stride_axes(input_shape: torch.Size, output_shape: torch.Size) -> tuple[int, ...]:
    return ()


class ReshapeOp(LinearOp):
    def __init__(self, input_shape: torch.Size, output_shape: tuple[int, ...]):
        input_shape = torch.Size(input_shape)
        if not isinstance(output_shape, torch.Size):
            output_shape = _meta_output_shape(lambda x: x.reshape(*output_shape), input_shape)
        self.target_shape = torch.Size(output_shape)
        tensor, input_dims, output_dims = _reshape_tensor(input_shape, self.target_shape)
        debug_jacobian = jacobian_from_function(
            input_shape,
            self.target_shape,
            lambda x: x.reshape(self.target_shape),
            zero_stride_axes=_reshape_zero_stride_axes(input_shape, self.target_shape),
        ) if DEBUG_LINEAR_OP else None
        super().__init__(
            tensor,
            input_dims,
            output_dims,
            flags=LinearOpFlags.IS_NON_NEGATIVE,
            debug_jacobian=debug_jacobian,
        )
        if DEBUG_LINEAR_OP:
            assert self.tensor.to_dense().allclose(self.debug_jacobian)

    def __str__(self):
        return f"<reshape {list(self.input_shape)} -> {list(self.target_shape)}>"


class FlattenOp(ReshapeOp):
    def __init__(self, input_shape: torch.Size, start_dim: int = 0, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim if end_dim >= 0 else len(input_shape) + end_dim
        self.original_sizes = torch.Size(input_shape)[self.start_dim:self.end_dim + 1]
        target_shape = _meta_output_shape(lambda x: x.flatten(start_dim, end_dim), torch.Size(input_shape))
        super().__init__(torch.Size(input_shape), target_shape)

    def __str__(self):
        return f"<flatten {self.start_dim} {self.end_dim}>"


class UnflattenOp(ReshapeOp):
    def __init__(self, input_shape: torch.Size, dim: int, sizes: tuple[int, ...]):
        self.dim = dim
        self.sizes = tuple(sizes)
        self.end_dim = dim + len(sizes) - 1
        target_shape = _meta_output_shape(lambda x: x.unflatten(dim, sizes), torch.Size(input_shape))
        super().__init__(torch.Size(input_shape), target_shape)

    def __str__(self):
        return f"<unflatten {self.dim} {list(self.sizes)}>"


class SqueezeOp(ReshapeOp):
    def __init__(self, input_shape: torch.Size, dim=None):
        input_shape = torch.Size(input_shape)
        self.dim = dim
        if dim is not None:
            self._is_noop = input_shape[dim] != 1
            target_shape = input_shape if self._is_noop else torch.Size(
                s for i, s in enumerate(input_shape) if i != dim
            )
        else:
            self._is_noop = all(s != 1 for s in input_shape)
            self._squeezed_dims = [i for i, s in enumerate(input_shape) if s == 1]
            target_shape = torch.Size(s for s in input_shape if s != 1)
        super().__init__(input_shape, target_shape)

    def __str__(self):
        return f"<squeeze {self.dim}>"


class UnsqueezeOp(ReshapeOp):
    def __init__(self, input_shape: torch.Size, dim: int):
        if dim < 0:
            dim += len(input_shape) + 1
        assert 0 <= dim <= len(input_shape)
        self.dim = dim
        target_shape = _meta_output_shape(lambda x: x.unsqueeze(dim), torch.Size(input_shape))
        super().__init__(torch.Size(input_shape), target_shape)

    def __str__(self):
        return f"<unsqueeze {list(self.input_shape)} {self.dim}>"
