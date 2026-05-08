"""Helpers for constructing sparse LinearOp tensors."""

from collections.abc import Callable

import torch

from boundlab.sparse import coo
from boundlab.sparse.coo import COOSparsify, MultiCOOSparsify, MultiCOOTensor, MultiCOOTensorSum
from boundlab.sparse.dim import Dim
from boundlab.sparse.table import TorchTable
from boundlab.sparse.tn import Dense, TN
from boundlab.linearop._base import LinearOp as BaseLinearOp, LinearOpFlags


def prod(shape: torch.Size) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def make_input_dims(shape: torch.Size) -> list[Dim]:
    return [Dim(size, 1000.0 + i, f"i{i}") for i, size in enumerate(shape)]


def make_output_dims(shape: torch.Size) -> list[Dim]:
    return [Dim(size, float(i), f"o{i}") for i, size in enumerate(shape)]


def unravel(flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    flat = flat.to(torch.long)
    if len(shape) == 0:
        return torch.empty((flat.numel(), 0), dtype=torch.long, device=flat.device)
    coords = []
    work = flat
    for size in reversed(shape):
        coords.append(work % int(size))
        work = work // int(size)
    return torch.stack(list(reversed(coords)), dim=1)


def ravel(coords: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if len(shape) == 0:
        return torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
    result = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
    for axis, size in enumerate(shape):
        result = result * int(size) + coords[:, axis].to(torch.long)
    return result


def all_coords(shape: torch.Size) -> torch.Tensor:
    return unravel(torch.arange(prod(shape), dtype=torch.long), shape)


def tensor_from_edges(
    input_shape: torch.Size,
    output_shape: torch.Size,
    input_coords: torch.Tensor,
    output_coords: torch.Tensor,
    values: torch.Tensor | None = None,
) -> tuple[MultiCOOTensorSum, list[Dim], list[Dim]]:
    input_shape = torch.Size(input_shape)
    output_shape = torch.Size(output_shape)
    input_dims = make_input_dims(input_shape)
    output_dims = make_output_dims(output_shape)
    edge_count = int(input_coords.shape[0])
    edge_dim = Dim(edge_count, 500.0, "e")
    if values is None:
        values = torch.ones(edge_count)
    values = values.reshape(edge_count).contiguous()

    columns = output_dims + input_dims
    data = []
    for axis in range(len(output_dims)):
        data.append(output_coords[:, axis].to(torch.long).contiguous())
    for axis in range(len(input_dims)):
        data.append(input_coords[:, axis].to(torch.long).contiguous())

    table = TorchTable(columns=columns, data=data, length=edge_count)
    op = COOSparsify(edge_dim, table)
    tensor = MultiCOOTensor(
        tn=TN.from_dense(Dense(values, [edge_dim])),
        sparsify=MultiCOOSparsify([op]),
    )
    return MultiCOOTensorSum([tensor]), input_dims, output_dims


def tensor_from_output_map(
    input_shape: torch.Size,
    output_shape: torch.Size,
    input_from_output: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[MultiCOOTensorSum, list[Dim], list[Dim]]:
    output_coords = all_coords(torch.Size(output_shape))
    input_coords = input_from_output(output_coords)
    return tensor_from_edges(input_shape, output_shape, input_coords, output_coords)
