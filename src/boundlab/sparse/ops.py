import pandas as pd
from typing import Union
import torch

from boundlab import utils

def table_join_sorted(*args: Union[torch.Tensor, list[int]]) -> torch.Tensor:
    """
    Join a list of tables on a common key.

    Args:
        $(
            tensor: torch.Tensor of shape (N, D_i) and dtype int8/16/32/64,
            columns: list[int] of length D_i, indicating the column indices in the output.
        )*

    Returns:
        torch.Tensor of shape (M, C) where M is the number of joined rows and C is the
        total number of output columns.
    """
    tensors: list[torch.Tensor] = [args[i * 2] for i in range(len(args) // 2)]
    columns: list[list[int]] = [args[i * 2 + 1] for i in range(len(args) // 2)]

    assert len(tensors) == len(columns)
    assert all(t.dim() == 2 and t.dtype in [torch.int8, torch.int16, torch.int32, torch.int64] for t in tensors)
    assert all(len(dims) == t.shape[1] for t, dims in zip(tensors, columns))
    if utils.current_fake_mode():
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        return torch.onnx.ops.symbolic(
            "boundlab::TableJoinSorted",
            inputs=args,
            dtype=tensors[0].dtype,
            shape=(ShapeEnv.create_symintnode(), ShapeEnv.create_symintnode()),
        )
    else:
        df = [
            pd.DataFrame(
                {f"t{columns[i][j]}": tensors[i][:, j].numpy() for j in range(tensors[i].shape[1])}
            )
            for i in range(len(tensors))
        ]

        result = df[0]
        for i in range(1, len(df)):
            shared = list(set(result.columns) & set(df[i].columns))
            if shared:
                result = result.merge(df[i], on=shared, how="inner")
            else:
                result = result.merge(df[i], how="cross")
        N = max(max(column) for column in columns) + 1
        cols = [f"t{i}" for i in range(N)]
        if result.empty:
            return torch.zeros((0, N), dtype=tensors[0].dtype)
        return torch.tensor(result[cols].values, dtype=tensors[0].dtype)

def list_index_unique(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Parallel of [list(tensor1).index(t) for t in tensor2] for tensors. Assumes that each row of tensor2 appears exactly once in tensor1.
    """
    # assert tensor1.dtype == tensor2.dtype
    # if utils.current_fake_mode():
    #     from torch.fx.experimental.symbolic_shapes import ShapeEnv
    #     return torch.onnx.ops.symbolic(
    #         "boundlab::ListIndexUnique",
    #         inputs=(tensor1, tensor2),
    #         dtype=torch.int64,
    #         shape=(tensor2.shape[0],),
    #     )
    # else:
    N = tensor1.shape[0]
    M = tensor2.shape[0]
    tensor1 = tensor1.reshape(N, -1)
    tensor2 = tensor2.reshape(M, -1)
    eq = (tensor1.unsqueeze(0) == tensor2.unsqueeze(1)).all(dim=2)  # [M, N]
    idx = torch.nonzero(eq, as_tuple=True)[1]
    assert idx.shape[0] == M, "Each row of tensor2 should appear exactly once in tensor1"
    return idx


def list_is_subset(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """
    Check if t1 is a subset of t2. Each row of t1 should appear at least once in t2.
    """
    
    N = t2.shape[0]
    M = t1.shape[0]
    t2 = t2.reshape(N, -1)
    t1 = t1.reshape(M, -1)
    return (t2.unsqueeze(0) == t1.unsqueeze(1)).all(dim=2).any(dim=1).all().item()


__all__ = ["table_join_sorted", "list_index_unique", "list_is_subset"]
