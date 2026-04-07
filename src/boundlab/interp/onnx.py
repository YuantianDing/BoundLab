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


# ---------------------------------------------------------------------------
# Global registry for custom ONNX op translations
# ---------------------------------------------------------------------------

# Maps torch op overload → onnxscript sentinel function
_ONNX_CUSTOM_TABLE: dict[Callable, Callable] = {}
# (sentinel_op_type, real_domain, real_op_type) triples applied after export
_ONNX_SENTINEL_FIXUPS: list[tuple[str, str, str]] = []


def register_onnx_translation(
    torch_op: Callable,
    sentinel_fn: Callable,
    domain: str,
    op_type: str,
) -> None:
    """Register a custom ONNX translation for a torch op.

    During :func:`onnx_export` the *sentinel_fn* (an onnxscript function) is
    used as a temporary placeholder.  After export the placeholder nodes are
    replaced in-place with a primitive custom node using *domain* / *op_type*.

    Args:
        torch_op: The ``torch.ops.<ns>.<name>.default`` overload to translate.
        sentinel_fn: An onnxscript function used as a temporary stand-in.
        domain: ONNX domain for the real custom node (e.g. ``"boundlab"``).
        op_type: ONNX op_type for the real custom node (e.g. ``"diff_pair"``).
    """
    _ONNX_CUSTOM_TABLE[torch_op] = sentinel_fn
    _ONNX_SENTINEL_FIXUPS.append((sentinel_fn.__name__, domain, op_type))


def _apply_sentinel_fixups(model: ir.Model) -> None:
    """Replace sentinel placeholder nodes with real custom-domain ONNX nodes.

    Raises:
        AssertionError: If any sentinel node remains in the graph after
            replacement (indicates a sentinel/fixup name mismatch).
    """
    if not _ONNX_SENTINEL_FIXUPS:
        return
    sentinel_map = {
        name: (domain, op_type)
        for name, domain, op_type in _ONNX_SENTINEL_FIXUPS
    }
    graph = model.graph
    to_replace = [n for n in graph if n.op_type in sentinel_map]
    for node in to_replace:
        domain, op_type = sentinel_map[node.op_type]
        custom = ir.Node(
            domain=domain,
            op_type=op_type,
            inputs=list(node.inputs),
            num_outputs=len(node.outputs),
        )
        graph.insert_before(node, custom)
        for old_val, new_val in zip(node.outputs, custom.outputs):
            new_val.name = old_val.name
            new_val.shape = old_val.shape
            new_val.dtype = old_val.dtype
            for use in list(old_val.uses()):
                use.node.replace_input_with(use.idx, new_val)
        graph.remove(node, safe=False)

    # Verify: no sentinel stubs should remain.
    remaining = [n.op_type for n in graph if n.op_type in sentinel_map]
    assert not remaining, (
        f"onnx_export: sentinel nodes were not fully replaced: {remaining}"
    )


def onnx_export(
    f: Callable[..., torch.Tensor] | nn.Module,
    args: tuple[Union[torch.Size, list[int]], ...],
    kwargs: dict[str, Union[torch.Size, list[int]]] = {},
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
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
        f = Wrapper().eval()
    else:
        f = f.eval()
    args_tensor = tuple(torch.zeros(s) for s in args)
    with FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(allow_dynamic_output_shape_ops=True),
    ):
        program = torch.export.export(f, args_tensor)
    # Determine which registered custom ops appear in the FX graph so we can
    # verify they survive into the ONNX output after fixups.
    registered_targets = set(_ONNX_CUSTOM_TABLE)
    fx_custom_ops = {
        node.target
        for node in program.graph.nodes
        if node.op == "call_function" and node.target in registered_targets
    }

    onnx_program = torch.onnx.export(
        program,
        args=(),
        export_params=True,
        optimize=not _ONNX_CUSTOM_TABLE,
        custom_translation_table=dict(_ONNX_CUSTOM_TABLE) if _ONNX_CUSTOM_TABLE else None,
        input_names=input_names,
        output_names=output_names,
    )
    model = onnx_program.model
    _apply_sentinel_fixups(model)

    # Verify that every custom op present in the FX graph appears in the ONNX
    # model as a node with the expected (domain, op_type).
    if fx_custom_ops:
        graph_ops = {(n.domain, n.op_type) for n in model.graph}
        for torch_op in fx_custom_ops:
            expected = _op_identity(torch_op)
            if expected is not None:
                assert expected in graph_ops, (
                    f"onnx_export: expected custom op {expected} in ONNX graph "
                    f"(registered for {torch_op}) but found only: {graph_ops}"
                )

    return model


def _op_identity(torch_op: Callable) -> tuple[str, str] | None:
    """Return (domain, op_type) registered for *torch_op*, or None."""
    sentinel_fn = _ONNX_CUSTOM_TABLE.get(torch_op)
    if sentinel_fn is None:
        return None
    for sentinel_name, domain, op_type in _ONNX_SENTINEL_FIXUPS:
        if sentinel_name == sentinel_fn.__name__:
            return (domain, op_type)
    return None
