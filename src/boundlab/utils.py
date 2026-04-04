r"""Utility Functions for BoundLab

This module provides helper functions used throughout the BoundLab framework.

Examples
--------
>>> from boundlab.utils import merge_name
>>> merge_name("x", "+", "y")
'(x + y)'
"""

import string
from typing import Callable, Literal, Sequence, TypeAlias, TypeVar, Union

from torch import nn
import onnx_ir as ir

A = TypeVar("A")

Triple: TypeAlias = tuple[A, A, A]
import torch
import tempfile
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

    
def merge_name(name1, op: str, name2) -> str | None:
    """Merge two optional names into a single name for a composed operation.

    Examples
    --------
    >>> merge_name("left", "@", "right")
    '(left @ right)'
    >>> merge_name(None, "@", "right") is None
    True
    """
    name1 = name1.name if hasattr(name1, "name") else name1
    name2 = name2.name if hasattr(name2, "name") else name2
    if type(name1) is not str or type(name2) is not str:
        return None
    if name1 is not None and name2 is not None:
        return f"({name1} {op} {name2})"
    return None

def onnx_export(
        f: Callable[..., torch.Tensor] | nn.Module,
        args: tuple[Union[torch.Size, list[int]], ...],
        kwargs: dict[str, Union[torch.Size, list[int]]] = {},
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
    ) -> ir.Model:
    """Export a PyTorch function to ONNX format.

    Examples
    --------
    >>> import torch
    >>> from boundlab.utils import onnx_export
    >>> def f(x):
    ...     return x @ x.T
    ...
    >>> model_proto = onnx_export(f, [3, 4])
    >>> list(model_proto.graph)[0].op_type
    'MatMul'
    """
    if not isinstance(f, nn.Module):
        class Wrapper(nn.Module):
            def forward(self, *args, **kwargs):
                return f(*args, **kwargs)
        f = Wrapper()
    elif isinstance(f, nn.Module):
        f = f.eval()
    args_tensor = tuple(torch.zeros(s) for s in args)
    program = torch.onnx.export(
        f, args_tensor,
        export_params=True, input_names=input_names, output_names=output_names)
    return program.model
            
