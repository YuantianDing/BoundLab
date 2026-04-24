"""δ-Top-1 equivalence verification via LP, following VeryDiff (Teuber et al.).

Given the output triple (Z_x, Z_y, Z_Δ) from differential propagation, verify
that for all inputs where f1's top class has softmax confidence ≥ δ, f2 agrees
on the top class.

The confidence constraint is linearized in logit space (VeryDiff Lemma 4):
    softmax(z)_k ≥ δ  ⟹  z_k - z_j ≥ t  for all j ≠ k
where t = ln(δ / (1 - δ)).

The LP (VeryDiff Definition 5): for each class pair (k, j):
    max  Z''_j(ε) - Z''_k(ε)
    s.t. Z'_k(ε) - Z'_l(ε) ≥ t   for all l ≠ k
         Z'(ε) = Z''(ε) + Z^Δ(ε)  (structural, from shared generators)
         ε ∈ [-1, 1]^n

If max ≤ 0 for all (k, j), then δ-Top-1 is certified.

Usage:
    from boundlab.diff.zono3 import interpret as diff_interpret
    from delta_top1 import verify_delta_top1, certify_sweep

    out = op(DiffExpr3(x, y, x - y))
    certified, details = verify_delta_top1(out, delta=0.9)
"""

import math
from collections import OrderedDict

import torch
import numpy as np
from scipy.optimize import linprog

from boundlab.expr._core import Expr
from boundlab.expr._affine import AffineSum, ConstVal
from boundlab.expr._var import LpEpsilon


# =====================================================================
# Step 1: Extract affine structure from Expr
# =====================================================================

def _collect_epsilons(*exprs: Expr) -> OrderedDict:
    """Collect all LpEpsilon nodes from one or more expressions.

    Returns an OrderedDict mapping epsilon.id -> (epsilon, start_col)
    where start_col is the starting column index in the unified
    generator matrix.
    """
    eps_list = OrderedDict()  # id -> LpEpsilon
    visited = set()

    def _walk(e):
        if id(e) in visited:
            return
        visited.add(id(e))
        if isinstance(e, LpEpsilon):
            if e.id not in eps_list:
                eps_list[e.id] = e
        if hasattr(e, 'children'):
            for child in e.children:
                _walk(child)

    for expr in exprs:
        _walk(expr)

    # Assign column ranges
    result = OrderedDict()
    col = 0
    for eid, eps in eps_list.items():
        result[eid] = (eps, col)
        col += eps.shape.numel()

    return result


def extract_affine(expr: Expr, eps_map: OrderedDict) -> tuple:
    """Extract (center, generator_matrix) from an affine Expr.

    Args:
        expr: An AffineSum expression (output of propagation).
        eps_map: OrderedDict from _collect_epsilons, mapping
            epsilon.id -> (LpEpsilon, start_col).

    Returns:
        center: 1D tensor of shape [output_flat].
        G: 2D tensor of shape [output_flat, total_eps_dims].
    """
    output_numel = expr.shape.numel()
    total_eps = sum(eps.shape.numel() for eps, _ in eps_map.values())

    # Center: constant term
    if isinstance(expr, ConstVal):
        c = expr.get_const()
        if isinstance(c, int) and c == 0:
            center = torch.zeros(output_numel)
        else:
            center = c.flatten()
        G = torch.zeros(output_numel, total_eps)
        return center, G

    assert isinstance(expr, AffineSum), f"Expected AffineSum, got {type(expr)}"

    center = expr.constant.flatten().clone() if expr.constant is not None \
        else torch.zeros(output_numel)

    G = torch.zeros(output_numel, total_eps)

    for child, op in expr.children_dict.items():
        assert isinstance(child, LpEpsilon), \
            f"Expected flattened AffineSum with LpEpsilon leaves, got {type(child)}. " \
            f"Call expr.simplify_ops_() first or ensure the expression is flat."

        eps_numel = child.shape.numel()
        _, start_col = eps_map[child.id]

        # Materialize the Jacobian: shape [*output_shape, *input_shape]
        jac = op.jacobian()
        # Flatten to [output_flat, input_flat]
        jac_flat = jac.reshape(output_numel, eps_numel)

        G[:, start_col:start_col + eps_numel] = jac_flat

    return center, G


# =====================================================================
# Step 2: Solve the LP
# =====================================================================

def _solve_top1_lp(c_x, G_x, c_y, G_y, k, j, t, O):
    """Solve one Top-1 Violation LP for class pair (k, j).

    maximize: Z''_j(ε) - Z''_k(ε)
    subject to: Z'_k(ε) - Z'_l(ε) ≥ t  for all l ≠ k
                ε ∈ [-1, 1]^n

    Returns the optimal value (positive = violation possible).
    """
    N = G_x.shape[1]  # number of epsilon dimensions

    # Objective: maximize (G_y[j] - G_y[k]) @ ε + (c_y[j] - c_y[k])
    # linprog minimizes, so negate
    obj = -(G_y[j] - G_y[k]).numpy()

    # Constraints: Z'_k(ε) - Z'_l(ε) ≥ t for all l ≠ k
    # i.e., (G_x[k] - G_x[l]) @ ε + (c_x[k] - c_x[l]) ≥ t
    # i.e., -(G_x[k] - G_x[l]) @ ε ≤ (c_x[k] - c_x[l]) - t
    A_ub_rows = []
    b_ub_rows = []

    for l in range(O):
        if l == k:
            continue
        row = -(G_x[k] - G_x[l]).numpy()
        rhs = (c_x[k] - c_x[l]).item() - t
        A_ub_rows.append(row)
        b_ub_rows.append(rhs)

    A_ub = np.stack(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # Bounds: ε ∈ [-1, 1]
    bounds = [(-1.0, 1.0)] * N

    result = linprog(
        c=obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs',
    )

    if not result.success:
        # Infeasible means no input makes f1 classify as k with margin t.
        # This means δ-Top-1 is vacuously true for this k.
        return -float('inf')

    # Optimal value of the original maximization
    opt_val = -result.fun + (c_y[j] - c_y[k]).item()
    return opt_val


# =====================================================================
# Step 3: Full δ-Top-1 verification
# =====================================================================

def verify_delta_top1(out, delta: float, verbose: bool = False):
    """Verify δ-Top-1 equivalence from differential propagation output.

    Args:
        out: DiffExpr3 with out.x (f1 output), out.y (f2 output),
             out.diff (differential). Each has shape [O].
        delta: Confidence threshold in (0.5, 1.0).
        verbose: Print per-pair LP results.

    Returns:
        certified: bool, True if δ-Top-1 is certified.
        worst_violation: float, maximum LP objective value across
            all (k, j) pairs. Negative means certified.
    """
    from boundlab.diff.expr import DiffExpr3
    assert isinstance(out, DiffExpr3)
    assert 0.5 < delta < 1.0, f"delta must be in (0.5, 1.0), got {delta}"

    t = math.log(delta / (1.0 - delta))
    O = out.x.shape[0]  # number of output classes

    # Collect all epsilon variables from both expressions
    eps_map = _collect_epsilons(out.x, out.y)

    if verbose:
        total_eps = sum(eps.shape.numel() for eps, _ in eps_map.values())
        print(f"  δ={delta}, t={t:.4f}, O={O}, N_eps={total_eps}")

    # Extract affine structure
    c_x, G_x = extract_affine(out.x, eps_map)
    c_y, G_y = extract_affine(out.y, eps_map)

    worst = -float('inf')

    for k in range(O):
        for j in range(O):
            if k == j:
                continue

            val = _solve_top1_lp(c_x, G_x, c_y, G_y, k, j, t, O)

            if verbose:
                status = "VIOLATION" if val > 0 else "ok"
                print(f"    k={k}, j={j}: LP val={val:.6f} [{status}]")

            worst = max(worst, val)

            if val > 0 and not verbose:
                # Early exit: violation found
                return False, val

    certified = worst <= 0
    return certified, worst


# =====================================================================
# Step 4: Sweep over delta values (VeryDiff Table 2 style)
# =====================================================================

def certify_sweep(
    model,
    centers: list,
    eps: float,
    deltas: list = None,
    verbose: bool = False,
):
    """Run δ-Top-1 verification over a dataset, sweeping delta.

    Args:
        model: nn.Module (the model to verify, with diff_pair or
               heaviside_pruning operators embedded).
        centers: List of input center tensors.
        eps: L∞ input perturbation radius.
        deltas: List of confidence thresholds to sweep.
            Default: [0.9, 0.95, 0.99, 0.999, 0.9999].
        verbose: Print per-sample details.

    Returns:
        results: dict mapping delta -> count of certified samples.
        per_sample: list of dicts with per-sample details.
    """
    import boundlab.expr as expr
    import boundlab.prop as prop
    from boundlab.diff.expr import DiffExpr3
    from boundlab.diff.zono3 import interpret as diff_interpret
    from boundlab.interp.onnx import onnx_export

    if deltas is None:
        deltas = [0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999]

    # Export and build interpreter once
    sample_shape = list(centers[0].shape)
    onnx_model = onnx_export(model, (sample_shape,))
    op = diff_interpret(onnx_model)

    results = {d: 0 for d in deltas}
    per_sample = []

    for i, center in enumerate(centers):
        prop._UB_CACHE.clear()
        prop._LB_CACHE.clear()

        # Build zonotope input
        eps_sym = expr.LpEpsilon(sample_shape)
        x = expr.ConstVal(center) + eps * eps_sym

        # Propagate
        out = op(x)

        if not isinstance(out, DiffExpr3):
            # Fallback: model didn't produce DiffExpr3
            if verbose:
                print(f"  Sample {i}: not a DiffExpr3, skipping")
            per_sample.append({'sample': i, 'certified_deltas': []})
            continue

        # Find best (lowest) delta that certifies
        sample_info = {'sample': i, 'certified_deltas': []}

        for delta in sorted(deltas):
            certified, worst_val = verify_delta_top1(out, delta, verbose=False)

            if certified:
                results[delta] += 1
                sample_info['certified_deltas'].append(delta)

            if verbose:
                status = "CERT" if certified else "FAIL"
                print(f"  Sample {i}, δ={delta}: {status} (worst={worst_val:.6f})")

        # Compute min certifiable delta via binary search
        lo, hi = 0.5 + 1e-9, 1.0 - 1e-9
        for _ in range(30):
            mid = (lo + hi) / 2
            cert, _ = verify_delta_top1(out, mid, verbose=False)
            if cert:
                hi = mid  # can certify at lower delta
            else:
                lo = mid
        sample_info['min_delta'] = lo
        per_sample.append(sample_info)

    return results, per_sample


# =====================================================================
# Step 5: Reporting (VeryDiff Table 2 format)
# =====================================================================

def print_report(results: dict, N: int, method_name: str = "Differential"):
    """Print a VeryDiff-style certified count table.

    Args:
        results: dict mapping delta -> certified count.
        N: total number of samples.
        method_name: name for the column header.
    """
    print(f"\n{'δ':>10} {'t=ln(δ/(1-δ))':>15} {method_name:>15} {'Rate':>10}")
    print("-" * 55)
    for delta in sorted(results.keys()):
        t = math.log(delta / (1.0 - delta))
        count = results[delta]
        rate = count / N * 100
        print(f"{delta:>10.6f} {t:>15.4f} {count:>10d}/{N:<4d} {rate:>9.1f}%")


def print_comparison(
    all_results: dict,
    N: int,
    deltas: list = None,
):
    """Print a multi-method comparison table.

    Args:
        all_results: dict mapping method_name -> {delta -> count}.
        N: total number of samples.
        deltas: which deltas to show (default: all).
    """
    methods = list(all_results.keys())
    if deltas is None:
        deltas = sorted(set().union(*[r.keys() for r in all_results.values()]))

    # Header
    header = f"{'δ':>10} {'t':>8}"
    for m in methods:
        header += f" {m:>12}"
    print(header)
    print("-" * len(header))

    # Rows
    for delta in deltas:
        t = math.log(delta / (1.0 - delta))
        row = f"{delta:>10.4f} {t:>8.3f}"
        for m in methods:
            count = all_results[m].get(delta, 0)
            row += f" {count:>8d}/{N:<3d}"
        print(row)


# =====================================================================
# Convenience: run all three methods and compare
# =====================================================================

def full_comparison(
    model_or_onnx,
    centers: list,
    eps: float,
    deltas: list = None,
    verbose: bool = False,
):
    """Run Int-Sub, Zono-Sub, and Differential, report VeryDiff-style table.

    Args:
        model_or_onnx: nn.Module or onnx_ir.Model.
        centers: list of input center tensors.
        eps: L∞ input perturbation radius.
        deltas: confidence thresholds to sweep.

    Returns:
        all_results: dict of {method -> {delta -> count}}.
    """
    import boundlab.expr as expr
    import boundlab.prop as prop
    import boundlab.zono as zono
    from boundlab.diff.expr import DiffExpr3
    from boundlab.diff.zono3 import interpret as diff_interpret
    from boundlab.interp.onnx import onnx_export

    if deltas is None:
        deltas = [0.9, 0.95, 0.99, 0.999, 0.9999]

    N = len(centers)
    sample_shape = list(centers[0].shape)

    # Export once
    if not hasattr(model_or_onnx, 'graph'):
        onnx_model = onnx_export(model_or_onnx, (sample_shape,))
    else:
        onnx_model = model_or_onnx

    op_diff = diff_interpret(onnx_model)
    op_zono = zono.interpret(onnx_model)

    results_int = {d: 0 for d in deltas}
    results_zono = {d: 0 for d in deltas}
    results_diff = {d: 0 for d in deltas}

    for i, center in enumerate(centers):
        if verbose and i % 100 == 0:
            print(f"Sample {i}/{N}...")

        # --- Differential ---
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        eps_sym = expr.LpEpsilon(sample_shape)
        x = expr.ConstVal(center) + eps * eps_sym
        out_diff = op_diff(x)

        if isinstance(out_diff, DiffExpr3):
            eps_map_diff = _collect_epsilons(out_diff.x, out_diff.y)
            c_x_d, G_x_d = extract_affine(out_diff.x, eps_map_diff)
            c_y_d, G_y_d = extract_affine(out_diff.y, eps_map_diff)
        else:
            c_x_d = G_x_d = c_y_d = G_y_d = None

        # --- Zono-Sub ---
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        eps_sym1 = expr.LpEpsilon(sample_shape)
        x1 = expr.ConstVal(center) + eps * eps_sym1
        out_x = op_zono(x1)

        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        eps_sym2 = expr.LpEpsilon(sample_shape)
        x2 = expr.ConstVal(center) + eps * eps_sym2
        out_y = op_zono(x2)

        eps_map_zono = _collect_epsilons(out_x, out_y)
        c_x_z, G_x_z = extract_affine(out_x, eps_map_zono)
        c_y_z, G_y_z = extract_affine(out_y, eps_map_zono)

        # --- Int-Sub: use interval bounds directly ---
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        ub_x, lb_x = out_x.ublb()
        ub_y, lb_y = out_y.ublb()

        O = out_x.shape[0]

        for delta in deltas:
            t = math.log(delta / (1.0 - delta))

            # Int-Sub: worst case is max over (ub_y[j] - lb_y[k])
            # subject to lb_x[k] - ub_x[l] >= t for all l != k
            int_certified = True
            for k in range(O):
                margin_ok = all(
                    (lb_x[k] - ub_x[l]).item() >= t
                    for l in range(O) if l != k
                )
                if not margin_ok:
                    continue
                # f1 classifies as k with margin t. Check f2.
                for j in range(O):
                    if j == k:
                        continue
                    if (ub_y[j] - lb_y[k]).item() > 0:
                        int_certified = False
                        break
                if not int_certified:
                    break
            if int_certified:
                results_int[delta] += 1

            # Zono-Sub: LP
            if c_x_z is not None:
                val = max(
                    _solve_top1_lp(c_x_z, G_x_z, c_y_z, G_y_z, k, j, t, O)
                    for k in range(O) for j in range(O) if k != j
                )
                if val <= 0:
                    results_zono[delta] += 1

            # Differential: LP
            if c_x_d is not None:
                val = max(
                    _solve_top1_lp(c_x_d, G_x_d, c_y_d, G_y_d, k, j, t, O)
                    for k in range(O) for j in range(O) if k != j
                )
                if val <= 0:
                    results_diff[delta] += 1

    all_results = {
        'Int-Sub': results_int,
        'Zono-Sub': results_zono,
        'Differential': results_diff,
    }

    print_comparison(all_results, N, deltas)
    return all_results