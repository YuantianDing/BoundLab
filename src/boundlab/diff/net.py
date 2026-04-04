"""Build a paired ONNX model for differential interpretation.

``diff_net`` merges two structurally-identical ONNX models by inserting
``boundlab::diff_pair`` nodes that pair the weights from both networks before
each shared ``Gemm`` layer.  At concrete-tensor runtime the model is
equivalent to *net1*; when interpreted by
:data:`~boundlab.diff.zono3.interpret`, the ``diff_pair`` nodes create
:class:`~boundlab.diff.expr.DiffExpr2` expressions that propagate both
branches through all subsequent operations.
"""

from __future__ import annotations

import copy
from itertools import zip_longest
from pathlib import Path

import onnx_ir as ir


def _load_onnx(model) -> ir.Model:
    """Load an ONNX model from a path or pass through an existing onnx_ir.Model."""
    if isinstance(model, (str, Path)):
        return ir.load(str(model))
    if isinstance(model, ir.Model):
        return model
    raise TypeError(f"Expected path or onnx_ir.Model, got {type(model)}")


def _value_ref(name: str) -> ir.Value:
    return ir.Value(name=name)


def _new_initializer(name: str, source: ir.Value) -> ir.Value:
    return ir.Value(
        name=name,
        shape=copy.deepcopy(source.shape),
        type=copy.deepcopy(source.type),
        doc_string=source.doc_string,
        const_value=copy.deepcopy(source.const_value),
        metadata_props=copy.deepcopy(source.metadata_props),
    )


def _clone_node_detached(node: ir.Node) -> ir.Node:
    return ir.Node(
        node.domain,
        node.op_type,
        [(_value_ref(v.name) if v is not None else None) for v in node.inputs],
        attributes={k: copy.deepcopy(v) for k, v in node.attributes.items()},
        outputs=[ir.Value(name=v.name) for v in node.outputs],
        overload=node.overload,
        version=node.version,
        name=node.name,
        doc_string=node.doc_string,
        metadata_props=copy.deepcopy(node.metadata_props),
    )


def _node_input_name(node: ir.Node, idx: int) -> str:
    v = node.inputs[idx]
    assert v is not None and v.name is not None
    return v.name


def diff_net(
    net1: ir.Model | str | Path,
    net2: ir.Model | str | Path,
) -> ir.Model:
    """Merge two structurally-identical ONNX models by pairing initializer inputs.

    For each aligned node pair in ``net1``/``net2``, any input position where
    both sides consume initializers is replaced by a shared
    ``boundlab::diff_pair`` value in the merged graph. This generalizes beyond
    affine operators (e.g. ``Gemm`` / ``MatMul``) to any operator that reads
    parameters as initializers.

    At concrete runtime, ``diff_pair`` is a no-op that returns branch-1 values
    (net1 semantics). Under differential interpretation, the paired values are
    lifted into :class:`~boundlab.diff.expr.DiffExpr2`.

    Parameters
    ----------
    net1, net2:
        ONNX models with the same graph structure.  Each may be an
        :class:`onnx_ir.Model`, a ``str``, or a :class:`pathlib.Path`.

    Returns
    -------
    A merged :class:`onnx_ir.Model`.

    Examples
    --------
    >>> from boundlab.diff.net import diff_net
    >>> from pathlib import Path
    >>> nets = Path("compare/veridiff/examples/nets")
    >>> merged = diff_net(nets / "single_layer_ref.onnx", nets / "single_layer_alt.onnx")
    >>> any(n.domain == "boundlab" and n.op_type == "diff_pair" for n in merged.graph)
    True
    """
    net1 = _load_onnx(net1)
    net2 = _load_onnx(net2)
    merged = net1.clone()

    # Build initializer dicts (name -> Value).
    init1 = merged.graph.initializers
    init2 = net2.graph.initializers

    # Align nodes by topological index.
    nodes1 = list(merged.graph)
    nodes2 = list(net2.graph)
    if len(nodes1) != len(nodes2):
        raise ValueError(
            f"Networks have different numbers of nodes: {len(nodes1)} vs {len(nodes2)}"
        )

    new_nodes: list[ir.Node] = []
    new_initializer_entries: dict[str, ir.Value] = {}
    # Cache: net2 initializer name -> cloned initializer name in merged model.
    cloned_init2_name: dict[str, str] = {}
    # Cache: (net1 init name, net2 init name) -> paired value name.
    paired_name: dict[tuple[str, str], str] = {}

    def _paired_input_value(name1: str, name2: str) -> ir.Value:
        key = (name1, name2)
        if key in paired_name:
            return _value_ref(paired_name[key])

        if name2 not in cloned_init2_name:
            new_name = f"_diff_init2_{len(cloned_init2_name)}"
            new_initializer_entries[new_name] = _new_initializer(new_name, init2[name2])
            cloned_init2_name[name2] = new_name
        else:
            new_name = cloned_init2_name[name2]

        out_name = f"_diff_paired_{len(paired_name)}"
        paired_name[key] = out_name
        new_nodes.append(
            ir.Node(
                "boundlab",
                "diff_pair",
                [_value_ref(name1), _value_ref(new_name)],
                outputs=[ir.Value(name=out_name)],
            )
        )
        return _value_ref(out_name)

    for idx, (node, node2) in enumerate(zip(nodes1, nodes2)):
        if node.op_type != node2.op_type or node.domain != node2.domain:
            raise ValueError(
                f"Node mismatch at index {idx}: "
                f"({node.domain}::{node.op_type}) vs ({node2.domain}::{node2.op_type})"
            )

        if len(node.inputs) != len(node2.inputs):
            raise ValueError(
                f"Node input-arity mismatch at index {idx}: "
                f"{len(node.inputs)} vs {len(node2.inputs)}"
            )

        remapped_inputs: list[ir.Value | None] = []
        for inp1, inp2 in zip_longest(node.inputs, node2.inputs):
            if inp1 is None:
                remapped_inputs.append(None)
                continue

            name1 = inp1.name
            # Keep original input when net2 optional input is missing.
            if inp2 is None or inp2.name is None:
                remapped_inputs.append(_value_ref(name1))
                continue

            name2 = inp2.name
            if name1 in init1 and name2 in init2:
                remapped_inputs.append(_paired_input_value(name1, name2))
            else:
                remapped_inputs.append(_value_ref(name1))

        new_nodes.append(
            ir.Node(
                node.domain,
                node.op_type,
                remapped_inputs,
                attributes={k: copy.deepcopy(v) for k, v in node.attributes.items()},
                outputs=[ir.Value(name=o.name) for o in node.outputs],
                overload=node.overload,
                version=node.version,
                name=node.name,
                doc_string=node.doc_string,
                metadata_props=copy.deepcopy(node.metadata_props),
            )
        )

    # Rewrite graph in place on the cloned model.
    merged.graph.remove(list(merged.graph), safe=False)
    merged.graph.extend(new_nodes)
    merged.graph.initializers.update(new_initializer_entries)
    if paired_name:
        merged.graph.opset_imports["boundlab"] = max(merged.graph.opset_imports.get("boundlab", 0), 1)
        merged.opset_imports["boundlab"] = max(merged.opset_imports.get("boundlab", 0), 1)
    return merged


__all__ = ["diff_net"]
