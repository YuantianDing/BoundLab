"""ONNX export utilities for BoundLab.

Provides :func:`onnx_export` and :func:`register_onnx_translation` — tools for
converting PyTorch modules to ONNX IR models that the abstract interpretation
:class:`~boundlab.interp.Interpreter` can consume.

Custom ops (e.g. ``boundlab::diff_pair``) are handled via a two-step process:

1. An *onnxscript sentinel function* is registered as a placeholder for the
   custom torch op during ``torch.onnx.export``.
2. After export, :func:`_apply_sentinel_fixups` replaces every sentinel node
   with a proper primitive custom-domain ONNX node.

Call :func:`register_onnx_translation` from the module that defines the custom
torch op to hook into this mechanism.

Examples
--------
>>> import torch
>>> from boundlab.interp.onnx import onnx_export
>>> def f(x):
...     return x @ x.T
...
>>> model = onnx_export(f, ([3, 4],))
>>> list(model.graph)[0].op_type
'MatMul'
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import onnx_ir as ir
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

__all__ = ["onnx_export", "register_onnx_translation"]

def onnx_export(
    f: Callable[..., torch.Tensor] | nn.Module,
    args: tuple[Union[torch.Size, list[int], torch.Tensor], ...],
    kwargs: dict[str, Union[torch.Size, list[int]]] = {},
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    optimize: bool = None,
) -> ir.Model:
    """Export a PyTorch function or module to an ONNX IR model.

    Shape arguments are given as lists/tuples of ints — zero-value tensors are
    constructed internally and only used for tracing.

    Args:
        f: A callable or :class:`torch.nn.Module` to export.
        args: Input shapes, one per positional argument
              (e.g. ``([3, 4],)`` for a single rank-2 input).
        kwargs: Keyword-argument shapes (rarely needed).
        input_names: Optional names for the ONNX graph inputs.
        output_names: Optional names for the ONNX graph outputs.

    Returns:
        An :class:`onnx_ir.Model` ready for abstract interpretation.

    Examples
    --------
    >>> import torch
    >>> from boundlab.interp.onnx import onnx_export
    >>> def f(x):
    ...     return x @ x.T
    ...
    >>> model = onnx_export(f, ([3, 4],))
    >>> list(model.graph)[0].op_type
    'MatMul'
    """
    if not isinstance(f, nn.Module):
        class Wrapper(nn.Module):
            def forward(self, *args, **kwargs):
                return f(*args, **kwargs)
        mod = Wrapper().eval()
    else:
        mod = f.eval()
    args_tensor = tuple(x if isinstance(x, torch.Tensor) else torch.zeros(x) for x in args)
    
    with FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(allow_dynamic_output_shape_ops=True),
    ):
        program = torch.export.export(mod, args_tensor)


    onnx_program = torch.onnx.export(
        program,
        args=(),
        export_params=True,
        optimize=optimize,
        input_names=input_names,
        output_names=output_names,
        verbose=False,
    )
    return onnx_program.model