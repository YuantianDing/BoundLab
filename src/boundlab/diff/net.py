"""Build a paired FX graph for differential interpretation."""

from __future__ import annotations

import copy

import torch
import torch.fx
from torch import nn

from boundlab.diff.op import DiffLinear


def diff_net(
    net1: torch.fx.GraphModule,
    net2: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Merge two structurally-identical GraphModules by pairing their linear layers.

    Returns a new :class:`~torch.fx.GraphModule` where every linear layer is
    replaced by a :class:`~boundlab.diff.op.DiffLinear` that holds the weights
    from both networks.  At runtime (concrete tensors) the module is equivalent
    to *net1*.  When interpreted by
    :data:`~boundlab.diff.zono3.interpret`, the ``DiffLinear`` nodes produce
    :class:`~boundlab.diff.expr.DiffExpr2` expressions tracking both branches.

    Handles two graph styles:

    * **ATen-level** (from ``torch.export.export(...).module()``): identifies
      ``aten.linear.default`` or the ``aten.t + aten.addmm`` decomposition and
      replaces each group with a new ``call_module`` node targeting a
      freshly-added ``DiffLinear`` submodule named ``"diff_linear_0"``,
      ``"diff_linear_1"``, …
    * **Module-level** (from ``onnx2torch.convert(...)``): identifies
      ``call_module`` nodes whose submodule is ``nn.Linear`` and replaces each
      with a ``call_module`` node targeting a freshly-added ``DiffLinear``
      submodule (same naming scheme).

    Parameters
    ----------
    net1, net2:
        GraphModules with the same graph structure.

    Returns
    -------
    A merged :class:`~torch.fx.GraphModule`.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from boundlab.diff.net import diff_net
    >>> from boundlab.diff.op import DiffLinear
    >>> m1 = nn.Linear(4, 3)
    >>> m2 = nn.Linear(4, 3)
    >>> gm1 = torch.export.export(m1, (torch.zeros(4),)).module()
    >>> gm2 = torch.export.export(m2, (torch.zeros(4),)).module()
    >>> merged = diff_net(gm1, gm2)
    >>> dl_targets = [n.target for n in merged.graph.nodes if n.op == 'call_module' and str(n.target).startswith('diff_linear_')]
    >>> len(dl_targets)
    1
    >>> isinstance(merged.get_submodule(dl_targets[0]), DiffLinear)
    True
    """
    merged = copy.deepcopy(net1)

    merged_groups = _find_linear_groups(merged)
    net2_groups   = _find_linear_groups(net2)

    if merged_groups or net2_groups:
        _pair_aten_linears(merged, merged_groups, net2, net2_groups)
    else:
        mod_merged = [n for n in merged.graph.nodes
                      if n.op == "call_module"
                      and isinstance(merged.get_submodule(n.target), nn.Linear)]
        mod_net2   = [n for n in net2.graph.nodes
                      if n.op == "call_module"
                      and isinstance(net2.get_submodule(n.target), nn.Linear)]
        _pair_module_linears(merged, mod_merged, net2, mod_net2)

    merged.graph.eliminate_dead_code()
    merged.recompile()
    return merged


# =====================================================================
# Internal helpers
# =====================================================================

def _resolve_get_attr(gm: torch.fx.GraphModule, node) -> torch.Tensor | None:
    """Return the tensor for a *get_attr* FX node, or ``None``."""
    if not isinstance(node, torch.fx.Node) or node.op != "get_attr":
        return None
    obj = gm
    for part in node.target.split("."):
        obj = getattr(obj, part)
    return obj.detach().clone()


def _find_linear_groups(gm: torch.fx.GraphModule) -> list[tuple[str, torch.fx.Node]]:
    """Return ``(kind, primary_node)`` pairs for each linear computation.

    Supported patterns:

    * ``"linear"``: ``aten.linear.default(input, weight, bias?)``
    * ``"addmm"``: ``aten.addmm(bias, input, aten.t(weight))``
    * ``"mv_add"``: ``aten.add(aten.mv(weight, input), bias)`` — used for 1-D inputs.
    * ``"matmul_add"``: ``aten.add(aten.matmul(input, weight_t), bias)`` —
      common in ONNX-exported affine layers where ``weight_t`` has shape
      ``(in_features, out_features)``.
    """
    groups = []
    mv_consumed: set[torch.fx.Node] = set()
    matmul_consumed: set[torch.fx.Node] = set()

    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue

        if n.target is torch.ops.aten.linear.default:
            groups.append(("linear", n))

        elif n.target is torch.ops.aten.addmm.default:
            if len(n.args) >= 3:
                weight_t = n.args[2]
                if (isinstance(weight_t, torch.fx.Node)
                        and weight_t.op == "call_function"
                        and weight_t.target is torch.ops.aten.t.default):
                    groups.append(("addmm", n))

        elif n.target is torch.ops.aten.add.Tensor and len(n.args) >= 2:
            # Detect:
            # - aten.add(mv_node, bias_node) or aten.add(bias_node, mv_node)
            # - aten.add(matmul_node, bias_node) or aten.add(bias_node, matmul_node)
            a0, a1 = n.args[0], n.args[1]
            mv_node = None
            matmul_node = None
            if (isinstance(a0, torch.fx.Node)
                    and a0.op == "call_function"
                    and a0.target is torch.ops.aten.mv.default
                    and isinstance(a1, torch.fx.Node)
                    and a1.op == "get_attr"):
                mv_node = a0
            elif (isinstance(a1, torch.fx.Node)
                    and a1.op == "call_function"
                    and a1.target is torch.ops.aten.mv.default
                    and isinstance(a0, torch.fx.Node)
                    and a0.op == "get_attr"):
                mv_node = a1
            if mv_node is not None and mv_node not in mv_consumed:
                mv_consumed.add(mv_node)
                groups.append(("mv_add", n))

            if (isinstance(a0, torch.fx.Node)
                    and a0.op == "call_function"
                    and a0.target is torch.ops.aten.matmul.default
                    and isinstance(a1, torch.fx.Node)
                    and a1.op == "get_attr"):
                matmul_node = a0
            elif (isinstance(a1, torch.fx.Node)
                    and a1.op == "call_function"
                    and a1.target is torch.ops.aten.matmul.default
                    and isinstance(a0, torch.fx.Node)
                    and a0.op == "get_attr"):
                matmul_node = a1
            if matmul_node is not None and matmul_node not in matmul_consumed:
                matmul_consumed.add(matmul_node)
                groups.append(("matmul_add", n))

    return groups


def _extract_linear_params(
    gm: torch.fx.GraphModule,
    kind: str,
    node: torch.fx.Node,
) -> tuple:
    """Return ``(input_node, weight_tensor, bias_tensor | None)``."""
    if kind == "linear":
        # aten.linear.default(input, weight, bias?)
        input_node = node.args[0]
        w = _resolve_get_attr(gm, node.args[1] if len(node.args) > 1 else None)
        b = _resolve_get_attr(gm, node.args[2] if len(node.args) > 2 else None)
    elif kind == "addmm":
        # aten.addmm.default(bias, input, weight_t) where weight_t = aten.t(weight)
        b = _resolve_get_attr(gm, node.args[0])
        input_node = node.args[1]
        weight_t_node = node.args[2]  # result of aten.t(weight)
        w = _resolve_get_attr(gm, weight_t_node.args[0])
    elif kind == "mv_add":
        # mv_add: aten.add(aten.mv(weight, input), bias) or reversed
        a0, a1 = node.args[0], node.args[1]
        if (isinstance(a0, torch.fx.Node)
                and a0.op == "call_function"
                and a0.target is torch.ops.aten.mv.default):
            mv_node, bias_node = a0, a1
        else:
            mv_node, bias_node = a1, a0
        w = _resolve_get_attr(gm, mv_node.args[0])
        input_node = mv_node.args[1]
        b = _resolve_get_attr(gm, bias_node)
    else:
        # matmul_add: aten.add(aten.matmul(input, weight_t), bias) or reversed
        # Here weight_t has shape (in_features, out_features), so transpose to
        # match nn.Linear.weight shape (out_features, in_features).
        a0, a1 = node.args[0], node.args[1]
        if (isinstance(a0, torch.fx.Node)
                and a0.op == "call_function"
                and a0.target is torch.ops.aten.matmul.default):
            matmul_node, bias_node = a0, a1
        else:
            matmul_node, bias_node = a1, a0
        input_node = matmul_node.args[0]
        w_t = _resolve_get_attr(gm, matmul_node.args[1])
        w = None if w_t is None else w_t.transpose(0, 1)
        b = _resolve_get_attr(gm, bias_node)
    return input_node, w, b


def _make_diff_linear(w1, b1, w2, b2) -> DiffLinear:
    out_feat, in_feat = w1.shape
    has_bias = b1 is not None and b2 is not None
    fc1 = nn.Linear(in_feat, out_feat, bias=has_bias)
    fc2 = nn.Linear(in_feat, out_feat, bias=has_bias)
    with torch.no_grad():
        fc1.weight.copy_(w1)
        fc2.weight.copy_(w2)
        if has_bias:
            fc1.bias.copy_(b1)
            fc2.bias.copy_(b2)
    return DiffLinear(fc1, fc2)


def _pair_aten_linears(
    merged: torch.fx.GraphModule,
    merged_groups: list,
    net2: torch.fx.GraphModule,
    net2_groups: list,
) -> None:
    """Replace ATen-level linear nodes with DiffLinear call_module nodes."""
    if len(merged_groups) != len(net2_groups):
        raise ValueError(
            f"Networks have different numbers of linear layers: "
            f"{len(merged_groups)} vs {len(net2_groups)}"
        )
    for i, ((kind1, mn), (kind2, n2)) in enumerate(zip(merged_groups, net2_groups)):
        input_node, w1, b1 = _extract_linear_params(merged, kind1, mn)
        _,          w2, b2 = _extract_linear_params(net2,   kind2, n2)

        if w1 is None or w2 is None:
            raise RuntimeError(f"Cannot resolve weight tensor for linear layer {i}")

        dl = _make_diff_linear(w1, b1, w2, b2)
        module_name = f"diff_linear_{i}"
        merged.add_module(module_name, dl)

        with merged.graph.inserting_before(mn):
            new_node = merged.graph.call_module(module_name, args=(input_node,))
        mn.replace_all_uses_with(new_node)
        merged.graph.erase_node(mn)


def _pair_module_linears(
    merged: torch.fx.GraphModule,
    merged_nodes: list,
    net2: torch.fx.GraphModule,
    net2_nodes: list,
) -> None:
    """Replace ``nn.Linear`` call_module nodes with DiffLinear call_module nodes."""
    if len(merged_nodes) != len(net2_nodes):
        raise ValueError(
            f"Networks have different numbers of linear layers: "
            f"{len(merged_nodes)} vs {len(net2_nodes)}"
        )
    for i, (mn, n2) in enumerate(zip(merged_nodes, net2_nodes)):
        fc1 = copy.deepcopy(merged.get_submodule(mn.target))
        fc2 = copy.deepcopy(net2.get_submodule(n2.target))
        dl = DiffLinear(fc1, fc2)
        module_name = f"diff_linear_{i}"
        merged.add_module(module_name, dl)

        with merged.graph.inserting_before(mn):
            new_node = merged.graph.call_module(module_name, args=mn.args)
        mn.replace_all_uses_with(new_node)
        merged.graph.erase_node(mn)


__all__ = ["diff_net"]
