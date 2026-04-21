"""Branch-and-bound splitting for differential certification.

Refines `certify_differential` by recursively partitioning the input ε-ball
along selected dimensions, certifying each sub-box, and aggregating the
per-box bounds.

Soundness: if the input ε-ball is covered by a finite set of sub-boxes
$\{B_k\}$, and for each $B_k$ we have sound bounds $ub_k, lb_k$ on
$f_1(x) - f_2(x)$ for $x \in B_k$, then $\max_k ub_k$ and $\min_k lb_k$
are sound bounds over the whole ε-ball.

API:
    ub, lb = certify_differential_split(model1, model2, center, eps,
                                         budget=16, split_strategy='largest')

where `budget` is the total number of leaf sub-boxes (doubling each split),
and `split_strategy` picks which input dimension to split at each node:
    'largest'  — split the dimension with the largest remaining eps
                 (cheap; good default).
    'gradient' — split the dimension with the largest L1 contribution to
                 the output zonotope's width (tighter, but requires a
                 backward pass).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Literal

import torch

import boundlab.expr as expr
import boundlab.prop as prop
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export


# ---------------------------------------------------------------------------
# Core: certify one sub-box
# ---------------------------------------------------------------------------

def _certify_box(merged_graph, center, eps_tensor):
    """Certify differential bound on one sub-box.

    `center`: shape = input shape, the box's center.
    `eps_tensor`: same shape as center, per-dim eps (so box is
        [center - eps, center + eps] elementwise).

    Returns (ub, lb) on f_1 - f_2 over the box.
    """
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    op = diff_interpret(merged_graph)
    # Per-dim eps via elementwise scaling of LpEpsilon.
    x = expr.ConstVal(center) + eps_tensor * expr.LpEpsilon(list(center.shape))
    out = op(x)
    if isinstance(out, DiffExpr3):
        # Use the final-reset safety net: min of Z_Δ and Z_x - Z_y.
        sub = out.x - out.y
        ub_d, lb_d = out.diff.ublb()
        ub_s, lb_s = sub.ublb()
        ub = torch.minimum(ub_d, ub_s)
        lb = torch.maximum(lb_d, lb_s)
        return ub, lb
    else:
        d = out.x - out.y
        return d.ublb()


# ---------------------------------------------------------------------------
# Split-dimension selection
# ---------------------------------------------------------------------------

def _pick_split_largest(eps_tensor):
    """Return the flat index of the input dim with largest remaining eps."""
    return int(eps_tensor.flatten().argmax().item())


def _pick_split_gradient(merged_graph, center, eps_tensor):
    """Return the input dim whose perturbation most affects the output width.

    Uses a cheap estimate: re-certify with each dim's eps halved individually
    and pick the one whose half-width shrinks most. This is O(n_dims) forward
    passes per split-decision, expensive for large inputs; use 'largest' by
    default.

    For a lighter-weight version, subsamples a small random set of dims.
    """
    flat_eps = eps_tensor.flatten()
    n = flat_eps.numel()
    # Baseline width at current box
    ub0, lb0 = _certify_box(merged_graph, center, eps_tensor)
    w0 = (ub0 - lb0).max().item()
    # Subsample up to 16 dims with nonzero eps (full sweep is too slow)
    nonzero = torch.nonzero(flat_eps > 0, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return 0
    k = min(16, nonzero.numel())
    idxs = nonzero[torch.randperm(nonzero.numel())[:k]]
    best_idx = int(idxs[0].item())
    best_gain = -1.0
    for idx_t in idxs:
        idx = int(idx_t.item())
        trial_eps = flat_eps.clone()
        trial_eps[idx] /= 2
        trial_eps = trial_eps.reshape(eps_tensor.shape)
        ub, lb = _certify_box(merged_graph, center, trial_eps)
        w = (ub - lb).max().item()
        gain = w0 - w
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    return best_idx


# ---------------------------------------------------------------------------
# Branch-and-bound
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Box:
    """One leaf in the branch-and-bound tree. Ordered by -width so heappop
    returns the widest box (the best candidate to split next)."""
    neg_width: float
    center: torch.Tensor = field(compare=False)
    eps: torch.Tensor = field(compare=False)
    ub: torch.Tensor = field(compare=False)
    lb: torch.Tensor = field(compare=False)


def certify_differential_split(
    model1, model2, center, eps,
    budget: int = 16,
    split_strategy: Literal["largest", "gradient"] = "largest",
    width_threshold: float = 0.0,
    verbose: bool = False,
):
    """Branch-and-bound differential certification.

    Arguments
    ---------
    model1, model2 : nn.Module
    center : torch.Tensor, input center
    eps : float or torch.Tensor, L∞ radius (scalar) or per-dim vector
    budget : int, max number of leaf sub-boxes (doubling per split)
    split_strategy : 'largest' (cheap) or 'gradient' (expensive)
    width_threshold : stop splitting a box once its bound width is below this
    verbose : print per-split diagnostics

    Returns (ub, lb) — elementwise sound upper/lower bounds on f_1 - f_2
    over the whole input ε-ball.
    """
    # Pre-export once; diff_net is expensive and doesn't depend on center/eps.
    shape = list(center.shape)
    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))
    merged = diff_net(gm1, gm2)

    # Normalize eps to a tensor shaped like center.
    if not isinstance(eps, torch.Tensor):
        eps_tensor = torch.full_like(center, float(eps))
    else:
        eps_tensor = eps.expand_as(center).clone()

    # Certify the root box.
    ub, lb = _certify_box(merged, center, eps_tensor)
    w = (ub - lb).max().item()

    heap: list[_Box] = [_Box(-w, center.clone(), eps_tensor.clone(), ub, lb)]
    n_leaves = 1

    if verbose:
        print(f"  [split] budget={budget} strategy={split_strategy}")
        print(f"  [split] root: width={w:.4e}")

    while n_leaves < budget and heap:
        # Pop the widest box.
        box = heapq.heappop(heap)
        box_width = -box.neg_width

        # Stop-early: if this (widest) box is already below threshold, we're done.
        if box_width <= width_threshold:
            heapq.heappush(heap, box)
            break

        # Pick split dim.
        if split_strategy == "gradient":
            split_flat = _pick_split_gradient(merged, box.center, box.eps)
        else:
            split_flat = _pick_split_largest(box.eps)

        # Shift center ± eps_i / 2 along that dim, halve eps_i.
        eps_i = box.eps.flatten()[split_flat].item()
        if eps_i <= 0:
            # Nothing to split on this dim; put back and stop.
            heapq.heappush(heap, box)
            break

        delta = torch.zeros_like(box.center).flatten()
        delta[split_flat] = eps_i / 2
        delta = delta.reshape(box.center.shape)

        new_eps = box.eps.clone()
        new_eps_flat = new_eps.flatten()
        new_eps_flat[split_flat] = eps_i / 2
        new_eps = new_eps_flat.reshape(box.eps.shape)

        child_boxes = []
        for sign in (-1.0, +1.0):
            child_center = box.center + sign * delta
            ub_c, lb_c = _certify_box(merged, child_center, new_eps)
            w_c = (ub_c - lb_c).max().item()
            child_boxes.append(_Box(-w_c, child_center, new_eps.clone(), ub_c, lb_c))

        for cb in child_boxes:
            heapq.heappush(heap, cb)
        n_leaves += 1  # net: popped 1, pushed 2 → +1 leaves

        if verbose:
            widths = sorted([-b.neg_width for b in heap], reverse=True)
            print(f"  [split] n_leaves={n_leaves} "
                  f"split dim={split_flat} child widths={widths[0]:.3e}/{widths[1] if len(widths)>1 else 0:.3e} "
                  f"max heap width={widths[0]:.3e}")

    # Aggregate across all leaf boxes: elementwise max of ubs, min of lbs.
    ubs = torch.stack([b.ub for b in heap], dim=0)
    lbs = torch.stack([b.lb for b in heap], dim=0)
    ub = ubs.max(dim=0).values
    lb = lbs.min(dim=0).values

    if verbose:
        print(f"  [split] final n_leaves={len(heap)}, "
              f"max width={(ub - lb).max().item():.4e}")

    return ub, lb
