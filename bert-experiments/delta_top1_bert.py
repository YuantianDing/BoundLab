"""δ-Top-1 equivalence verification: BERT on SST-2 under weight quantization.

Loads the DeepT checkpoint, embeds SST-2 sentences, quantizes weights,
and certifies δ-Top-1 equivalence using three methods.

Reports VeryDiff-style tables: certified count per δ level per method.

Usage:
    python experiments/delta_top1_bert.py --layers 1 --bits 8
    python experiments/delta_top1_bert.py --layers 2 --bits 4 8 16
    python experiments/delta_top1_bert.py --layers 1 --bits 8 --n-samples 100
"""

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export
from boundlab.diff.delta_top1 import (
    _collect_epsilons, extract_affine, _solve_top1_lp, verify_delta_top1,
)

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Import model/data utilities from existing experiment
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys, os
sys.path.insert(0, str(HERE))

MODEL_DIR = ROOT / os.environ.get("BOUNDLAB_MODEL_DIR", "model")

from certify_sst2_bert import (
    DeepTBert, load_checkpoint, load_vocab, embed_sentence,
    load_sst2_samples, quantize_model,
)


# ---------------------------------------------------------------------------
# δ-Top-1 LP for a pair of Expr (not necessarily DiffExpr3)
# ---------------------------------------------------------------------------

def delta_top1_from_exprs(expr_x, expr_y, delta):
    """Run the δ-Top-1 LP on two output Exprs."""
    t = math.log(delta / (1.0 - delta))
    O = expr_x.shape[0]

    eps_map = _collect_epsilons(expr_x, expr_y)
    c_x, G_x = extract_affine(expr_x, eps_map)
    c_y, G_y = extract_affine(expr_y, eps_map)

    worst = -float('inf')
    for k in range(O):
        for j in range(O):
            if k == j:
                continue
            val = _solve_top1_lp(c_x, G_x, c_y, G_y, k, j, t, O)
            worst = max(worst, val)
    return worst <= 0, worst


# ---------------------------------------------------------------------------
# Three verification methods returning output expressions
# ---------------------------------------------------------------------------

def propagate_int_sub(model1, model2, center, eps):
    """Int-Sub: independent zonotopes for f1, f2."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    shape = list(center.shape)
    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))
    x1 = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out1 = zono.interpret(gm1)(x1)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    x2 = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out2 = zono.interpret(gm2)(x2)
    return out1, out2


def propagate_zono_sub(model1, model2, center, eps):
    """Zono-Sub: shared input zonotope for f1, f2."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    shape = list(center.shape)
    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out1 = zono.interpret(gm1)(x)
    out2 = zono.interpret(gm2)(x)
    return out1, out2


def propagate_differential(model1, model2, center, eps):
    """Differential: diff_net merged graph."""
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    shape = list(center.shape)
    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))
    merged = diff_net(gm1, gm2)
    op = diff_interpret(merged)
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out = op(x)
    if isinstance(out, DiffExpr3):
        return out.x, out.y
    else:
        return out, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--bits", type=int, nargs="+", default=[8])
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--deltas", type=float, nargs="+",
                        default=[0.6, 0.8, 0.9, 0.95, 0.99, 0.999])
    args = parser.parse_args()

    # Load
    print("Loading checkpoint...")
    vocab = load_vocab(MODEL_DIR / "vocab.txt")
    model1, state, config = load_checkpoint(num_layers=args.layers)

    print("Loading SST-2 samples...")
    samples = load_sst2_samples(model1, state, vocab,
                                n=args.n_samples, max_len=args.max_len)
    N = len(samples)
    print(f"Using {N} correctly classified samples")

    methods = [
        ("Int-Sub", propagate_int_sub),
        ("Zono-Sub", propagate_zono_sub),
        ("Differential", propagate_differential),
    ]

    for bits in args.bits:
        model2 = quantize_model(model1, bits)

        print(f"\n{'='*70}")
        print(f"  BERT {args.layers}L, {bits}-bit quant, eps={args.eps}, N={N}")
        print(f"{'='*70}")

        # Results: method -> delta -> count
        results = {name: {d: 0 for d in args.deltas} for name, _ in methods}
        # Also track d_bar per method for margin-based reporting
        all_dbar = {name: [] for name, _ in methods}
        times = {name: 0.0 for name, _ in methods}

        for i, (sent, center, tokens) in enumerate(samples):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{N}] {sent[:50]}...")

            for name, prop_fn in methods:
                t0 = time.perf_counter()
                try:
                    out_x, out_y = prop_fn(model1, model2, center, args.eps)

                    # Compute d_bar = max |ub_diff|, |lb_diff| for reference
                    d = out_x - out_y
                    d_ub, d_lb = d.ublb()
                    dbar = max(d_ub.abs().max().item(), d_lb.abs().max().item())
                    all_dbar[name].append(dbar)

                    # δ-Top-1 LP for each delta
                    for delta in args.deltas:
                        certified, _ = delta_top1_from_exprs(out_x, out_y, delta)
                        if certified:
                            results[name][delta] += 1

                except Exception as e:
                    all_dbar[name].append(float('inf'))

                elapsed = time.perf_counter() - t0
                times[name] += elapsed

        # Print VeryDiff-style table
        print(f"\n  {'δ':>8} {'t':>8}", end="")
        for name, _ in methods:
            print(f" {name:>14}", end="")
        print()
        print(f"  {'-'*8} {'-'*8}", end="")
        for _ in methods:
            print(f" {'-'*14}", end="")
        print()

        for delta in args.deltas:
            t = math.log(delta / (1.0 - delta))
            print(f"  {delta:>8.4f} {t:>8.3f}", end="")
            for name, _ in methods:
                c = results[name][delta]
                print(f" {c:>8d}/{N:<4d}", end="")
            print()

        # Avg d_bar and timing
        print(f"\n  {'Method':<15} {'Avg d_bar':>10} {'Time':>10}")
        print(f"  {'-'*38}")
        for name, _ in methods:
            db = all_dbar[name]
            finite = [x for x in db if x < float('inf')]
            avg = sum(finite) / len(finite) if finite else float('inf')
            print(f"  {name:<15} {avg:>10.6f} {times[name]:>9.1f}s")

        # Min certifiable delta per method (using margin check as proxy)
        print(f"\n  {'Method':<15} {'Min δ (mean)':>14} {'Min δ (worst)':>14}")
        print(f"  {'-'*45}")
        for name, _ in methods:
            db = all_dbar[name]
            finite = [x for x in db if x < float('inf')]
            if finite:
                # δ_min = sigmoid(2 * d_bar)
                d_mins = [1.0 / (1.0 + math.exp(-2 * d)) for d in finite]
                print(f"  {name:<15} {sum(d_mins)/len(d_mins):>14.6f} "
                      f"{max(d_mins):>14.6f}")


if __name__ == "__main__":
    main()