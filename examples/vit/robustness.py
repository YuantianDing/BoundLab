"""L∞ robustness certification for the MNIST ViT (1-layer and 3-layer).

Run::

    python robustness.py
    python robustness.py --method poly --model 3 --eps 0.003 --n-samples 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boundlab.expr as expr
import boundlab.zono as zono
import boundlab.poly as poly
from boundlab.interp import Interpreter
from boundlab.interp.onnx import onnx_export

from mnist_vit import mnist_vit, mnist_vit_3

METHODS: dict[str, Interpreter] = {
    "zono": zono.interpret,
    "poly": poly.interpret,
}


def load_test_samples(n: int, data_dir: str, seed: int):
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(
            data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(ds), generator=g)[:n].tolist()
        samples = [(ds[i][0], int(ds[i][1])) for i in indices]
        print(f"[data] loaded {len(samples)} real MNIST test samples")
        return samples
    except Exception as e:
        print(f"[data] WARNING: MNIST unavailable ({type(e).__name__}: {e}).")
        print("[data] Falling back to synthetic inputs (smoke test only).")
        g = torch.Generator().manual_seed(seed)
        return [(torch.rand(1, 28, 28, generator=g), -1) for _ in range(n)]

DEFAULT_DATA_DIR = Path(__file__).parent / "./mnist_data"
def certify(
    *,
    method: str = "zono",
    model: int = 1,
    eps: float = 0.005,
    n_samples: int = 20,
    seed: int = 0,
    data_dir: str = DEFAULT_DATA_DIR,
    input_norm: tuple[float, float] | None = (0.1307, 0.3081),
) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {list(METHODS)}")

    torch.manual_seed(seed)
    net = (mnist_vit(input_norm=input_norm) if model == 1
           else mnist_vit_3(input_norm=input_norm)).eval()

    print(f"[export] building {method} interpreter ...", flush=True)
    t0 = time.time()
    op = METHODS[method](onnx_export(net, ([1, 28, 28],)))
    print(f"[export] done in {time.time() - t0:.1f}s\n")

    samples = load_test_samples(n_samples, data_dir, seed)
    print()

    n_correct = n_certified = 0
    sum_max_width = 0.0
    hdr = f"{'#':>3} {'label':>5} {'pred':>4} {'correct':>7} {'margin':>10}  {'time':>6}  status"
    print(hdr)
    print("-" * len(hdr))

    for i, (img, label) in enumerate(samples):
        with torch.no_grad():
            pred = int(net(img).argmax().item())
        correct = (label >= 0) and (pred == label)
        n_correct += int(correct)

        t0 = time.time()
        x = expr.ConstVal(img) + eps * expr.LpEpsilon(list(img.shape))
        ub, lb = op(x).ublb()
        dt = time.time() - t0
        max_width = float((ub - lb).max().item())
        sum_max_width += max_width

        ub_others = ub.clone()
        ub_others[pred] = float("-inf")
        margin = float(lb[pred] - ub_others.max())
        certified = margin > 0.0
        n_certified += int(certified)

        tag = "CERT" if certified else "fail"
        lbl = f"{label:>5}" if label >= 0 else "    -"
        cor = "yes" if correct else ("  -" if label < 0 else " no")
        print(f"{i+1:>3} {lbl} {pred:>4} {cor:>7} {margin:>+10.4f} {dt:>5.1f}s  {tag}")

    print()
    print("=" * 58)
    print(f"  method         : {method}")
    print(f"  model          : mnist_vit (depth={model})")
    print(f"  eps (L∞ pixel) : {eps}")
    norm_str = f"(μ={input_norm[0]}, σ={input_norm[1]})" if input_norm else "off"
    print(f"  input_norm     : {norm_str}")
    print(f"  samples        : {n_samples}")
    if any(lbl >= 0 for _, lbl in samples):
        print(f"  clean accuracy : {n_correct}/{n_samples}"
              f" = {100*n_correct/n_samples:.1f}%")
    print(f"  certified      : {n_certified}/{n_samples}"
          f" = {100*n_certified/n_samples:.1f}%")
    print(f"  avg max width  : {sum_max_width / n_samples:.6f}")
    print("=" * 58)

    return {
        "n_total": n_samples,
        "n_correct": n_correct,
        "n_certified": n_certified,
        "avg_max_width": sum_max_width / n_samples,
    }


def _parse_input_norm(s: str) -> tuple[float, float] | None:
    if s.lower() == "none":
        return None
    mean, std = s.split(",")
    return float(mean), float(std)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BoundLab robustness certification for MNIST ViT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--method", nargs="+", default=list(METHODS),
                    choices=list(METHODS),
                    help="Verification method(s). Defaults to all.")
    ap.add_argument("--model", type=int, choices=[1, 3], default=1,
                    help="ViT depth: 1-layer or 3-layer.")
    ap.add_argument("--eps", type=float, default=0.005,
                    help="L∞ perturbation radius on pixel values.")
    ap.add_argument("--n-samples", type=int, default=20, dest="n_samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir")
    ap.add_argument("--input-norm", default="0.1307,0.3081", dest="input_norm",
                    help="Normalisation as 'mean,std', or 'none' to disable.")
    args = ap.parse_args()
    args.input_norm = _parse_input_norm(args.input_norm)
    methods = args.method

    print("=" * 58)
    print(f"  BoundLab Certification — MNIST ViT (depth={args.model})")
    print("=" * 58)
    print()

    results = {}
    for m in methods:
        print(f"[ {m} ]")
        results[m] = certify(**{**vars(args), "method": m})
        print()

    if len(methods) > 1:
        print("=" * 58)
        print("  Summary")
        print(f"  {'method':<8}  {'certified':>10}  {'avg max width':>14}")
        print("-" * 58)
        for m, r in results.items():
            print(f"  {m:<8}  {r['n_certified']:>4}/{r['n_total']:<4}"
                  f"  {r['avg_max_width']:>14.6f}")
        if "poly" in results and "zono" in results:
            delta = results["poly"]["avg_max_width"] - results["zono"]["avg_max_width"]
            print("-" * 58)
            print(f"  {'poly - zono':<8}  {'':>10}  {delta:>+14.6f}")
        print("=" * 58)


if __name__ == "__main__":
    main()
