"""Compare BoundLab vs alpha-beta-CROWN on ViT robustness certification.

Certifies L-infinity robustness on random CIFAR-10-shaped inputs around the
model's own top-1 prediction, running two tools on identical (ONNX, VNN-LIB)
inputs:

- **BoundLab**: zonotope bound propagation via ``boundlab.zono.interpret``.
- **alpha-beta-CROWN**: subprocess invocation of
  ``complete_verifier/abcrown.py`` with a generated VNN-LIB robustness spec.

The ViT under test is ``examples/vit/vit.py`` (``vit_ibp_3_3_8`` or
``vit_pgd_2_3_16``).

Usage::

    pixi run python compare/abcrown/compare.py

Requires an alpha-beta-CROWN checkout. Either set the environment variable
``ABCROWN_DIR`` to the repo root or clone it into
``compare/abcrown/alpha-beta-CROWN`` so that
``alpha-beta-CROWN/complete_verifier/abcrown.py`` is resolvable. If neither
is found, the abcrown column reports ``N/A`` and only BoundLab runs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import onnx_ir
import torch
import torch.nn as nn

# Make `examples/vit/vit.py` importable without modifying project-wide sys.path.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import boundlab.expr as expr  # noqa: E402
import boundlab.zono as zono  # noqa: E402
from boundlab.interp.onnx import onnx_export  # noqa: E402
from vit import vit as vit_module  # noqa: E402  # examples/vit/vit.py


NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)


# =====================================================================
# Batched ViT forward (for alpha-beta-CROWN which needs a leading batch dim)
# =====================================================================
#
# examples/vit/vit.py runs forward on a single (C,H,W) image, internally
# calling ``img.unsqueeze(0)``. abcrown passes inputs shaped (B,C,H,W);
# feeding them into the unbatched model would produce a 5-D tensor at the
# conv. We mirror the forward here over a leading batch dim so a properly
# batched ONNX can be exported specifically for abcrown. BoundLab keeps
# using the original unbatched ONNX exported by ``onnx_export``.

def _batched_attn(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # x: (B, N, D)
    B, N, _ = x.shape
    h = attn.heads
    q = attn.to_q(x).reshape(B, N, h, -1).permute(0, 2, 1, 3)  # (B, h, N, d)
    k = attn.to_k(x).reshape(B, N, h, -1).permute(0, 2, 1, 3)
    v = attn.to_v(x).reshape(B, N, h, -1).permute(0, 2, 1, 3)
    dots = (q @ k.transpose(-2, -1)) * attn.scale              # (B, h, N, N)
    w = dots.softmax(dim=-1)
    out = w @ v                                                 # (B, h, N, d)
    out = out.permute(0, 2, 1, 3).reshape(B, N, -1)            # (B, N, h*d)
    return attn.to_out(out)


class BatchedViT(nn.Module):
    """Mirror of ``examples.vit.vit.ViT.forward`` with a leading batch dim."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        m = self.model
        x = m.patch_conv(imgs)                # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        # Broadcast cls_token to (B, 1, D) via addition to avoid an Expand op
        # that bakes in the traced batch size.
        cls = torch.zeros_like(x[:, :1, :]) + m.cls_token
        x = torch.cat([cls, x], dim=1)        # (B, N+1, D)
        x = x + m.pos_embedding[: x.shape[1]]
        for block in m.blocks:
            x_attn = _batched_attn(block.attn, block.attn_norm(x)) + x
            x = block.ff(block.ff_norm(x_attn)) + x_attn
        x = x.mean(dim=1)                     # (B, D)
        return m.head_linear(m.head_norm(x))  # (B, num_classes)


# =====================================================================
# ONNX post-processing for onnx2pytorch (used by alpha-beta-CROWN)
# =====================================================================

def _patch_onnx_for_onnx2pytorch(onnx_path: Path) -> None:
    """Ensure Conv nodes carry an explicit ``kernel_shape`` attribute.

    BoundLab's exporter omits ``kernel_shape`` on Conv (it's optional in ONNX
    and inferable from the weight tensor), but abcrown's onnx2pytorch path
    requires it. We read the model, fill in ``kernel_shape`` from the weight
    initializer when missing, and overwrite the file in place.
    """
    import onnx  # local import — only needed when abcrown is enabled

    model = onnx.load(str(onnx_path))
    inits = {init.name: init for init in model.graph.initializer}
    modified = False
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(a.name == "kernel_shape" for a in node.attribute):
            continue
        if len(node.input) < 2 or node.input[1] not in inits:
            continue
        w_shape = list(inits[node.input[1]].dims)  # (out, in/g, *kernel)
        kernel_shape = w_shape[2:]
        if not kernel_shape:
            continue
        node.attribute.append(
            onnx.helper.make_attribute("kernel_shape", kernel_shape)
        )
        modified = True
    if modified:
        onnx.save(model, str(onnx_path))


# =====================================================================
# BoundLab certification
# =====================================================================

def boundlab_certify(
    op, center: torch.Tensor, predicted: int, eps: float
) -> tuple[bool, float]:
    """Return (certified, margin) where margin = lb[pred] - max_{j!=pred} ub[j]."""
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(list(center.shape))
    ub, lb = op(x).ublb()
    ub_others = ub.clone()
    ub_others[predicted] = float("-inf")
    margin = float(lb[predicted] - ub_others.max())
    return margin > 0.0, margin


# =====================================================================
# VNN-LIB robustness spec (abcrown convention: property violated ⇒ unsafe)
# =====================================================================

def write_vnnlib(
    path: Path,
    center: torch.Tensor,
    predicted: int,
    num_classes: int,
    eps: float,
    input_lo: float = 0.0,
    input_hi: float = 1.0,
) -> None:
    flat = center.flatten().tolist()
    lines: list[str] = [f"; L-inf robustness, eps={eps}, predicted={predicted}"]
    for i in range(len(flat)):
        lines.append(f"(declare-const X_{i} Real)")
    for j in range(num_classes):
        lines.append(f"(declare-const Y_{j} Real)")

    for i, v in enumerate(flat):
        lo = max(input_lo, v - eps)
        hi = min(input_hi, v + eps)
        lines.append(f"(assert (<= X_{i} {hi:.8f}))")
        lines.append(f"(assert (>= X_{i} {lo:.8f}))")

    # Robustness violated iff some other class's logit meets/exceeds predicted's.
    disjuncts = [
        f"(and (>= Y_{j} Y_{predicted}))"
        for j in range(num_classes) if j != predicted
    ]
    if len(disjuncts) == 1:
        lines.append(f"(assert {disjuncts[0]})")
    else:
        lines.append("(assert (or " + " ".join(disjuncts) + "))")

    path.write_text("\n".join(lines) + "\n")


# =====================================================================
# alpha-beta-CROWN runner
# =====================================================================

def find_abcrown_script() -> Path | None:
    candidates: list[Path] = []
    env = os.environ.get("ABCROWN_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(HERE / "alpha-beta-CROWN")
    for root in candidates:
        script = root / "complete_verifier" / "abcrown.py"
        if script.is_file():
            return script
    return None


_RESULT_PATTERNS = [
    # abcrown prints e.g. "Result: safe-incomplete" / "unsafe-pgd" / "unknown"
    re.compile(r"Result:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE),
    re.compile(r"verified\s+(safe|unsafe|unknown)", re.IGNORECASE),
    re.compile(r"^\s*(safe|unsafe|unknown|timeout)\s*$", re.IGNORECASE | re.MULTILINE),
]


def _extract_abcrown_result(output: str) -> str:
    for pat in _RESULT_PATTERNS:
        m = pat.search(output)
        if m:
            return m.group(1).lower()
    return "error"


_MINIMAL_CONFIG = """\
general:
  device: cpu
  conv_mode: matrix
"""


def _write_minimal_config(path: Path) -> None:
    path.write_text(_MINIMAL_CONFIG)


def abcrown_verify(
    abcrown_script: Path,
    onnx_path: Path,
    vnnlib_path: Path,
    config_path: Path,
    input_shape: tuple[int, ...],
    timeout: int,
) -> tuple[str, float]:
    """Invoke abcrown as a subprocess; return (result, elapsed_seconds)."""
    cmd = [
        sys.executable,
        str(abcrown_script),
        "--config", str(config_path),
        "--onnx_path", str(onnx_path),
        "--vnnlib_path", str(vnnlib_path),
        "--device", "cpu",
        "--input_shape", "-1", *[str(d) for d in input_shape],
        "--timeout", str(timeout),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout + 30,
            cwd=abcrown_script.parent.parent,
        )
    except FileNotFoundError:
        return "missing", time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return "timeout", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    output = (proc.stdout or "") + (proc.stderr or "")
    result = _extract_abcrown_result(output)
    if result == "error" and proc.returncode != 0:
        first = (proc.stderr or proc.stdout).strip().splitlines()
        first_line = first[0] if first else f"exit {proc.returncode}"
        print(f"  [abcrown error: {first_line}]", file=sys.stderr)
    return result, elapsed


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BoundLab vs alpha-beta-CROWN on ViT L-inf robustness."
    )
    parser.add_argument("--model", choices=["ibp_3_3_8", "pgd_2_3_16"],
                        default="ibp_3_3_8")
    parser.add_argument("--layer-norm", choices=["standard", "no_var"],
                        default="no_var", dest="layer_norm")
    parser.add_argument("--eps", type=float, default=0.002,
                        help="L-inf perturbation radius (default 0.002).")
    parser.add_argument("--n-samples", type=int, default=5, dest="n_samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--abcrown-timeout", type=int, default=120,
                        dest="abcrown_timeout")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ctor = {
        "ibp_3_3_8": vit_module.vit_ibp_3_3_8,
        "pgd_2_3_16": vit_module.vit_pgd_2_3_16,
    }[args.model]
    model = ctor(layer_norm_type=args.layer_norm).eval()

    tmp_dir = HERE / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"Exporting {args.model} to ONNX ...", flush=True)
    t0 = time.perf_counter()
    # BoundLab uses the unbatched model as-is.
    ir_model = onnx_export(model, (list(INPUT_SHAPE),))
    op = zono.interpret(ir_model)
    # abcrown uses a batched ONNX with a leading batch dim.
    onnx_path = tmp_dir / f"vit_{args.model}_batched.onnx"
    batched_model = BatchedViT(model).eval()
    with torch.no_grad():
        torch.onnx.export(
            batched_model,
            torch.zeros(1, *INPUT_SHAPE),
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
    _patch_onnx_for_onnx2pytorch(onnx_path)
    print(f"  export + compile: {time.perf_counter() - t0:.1f}s")

    abcrown_script = find_abcrown_script()
    abcrown_config: Path | None = None
    if abcrown_script is None:
        print(
            "[warn] alpha-beta-CROWN not found. Set ABCROWN_DIR or clone into "
            "compare/abcrown/alpha-beta-CROWN.",
            file=sys.stderr,
        )
    else:
        abcrown_config = tmp_dir / "abcrown_minimal.yaml"
        _write_minimal_config(abcrown_config)

    header = (
        f"{'Sample':<10} {'Pred':>4} {'BoundLab':>10} {'Margin':>10} "
        f"{'ABCROWN':>10} {'BL(s)':>8} {'AB(s)':>8}"
    )
    print()
    print(header)
    print("-" * len(header))

    bl_total = 0.0
    ab_total = 0.0
    bl_certified = 0
    ab_safe = 0

    for i in range(args.n_samples):
        center = torch.rand(*INPUT_SHAPE)
        with torch.no_grad():
            predicted = int(model(center).argmax().item())

        bl_start = time.perf_counter()
        cert, margin = boundlab_certify(op, center, predicted, args.eps)
        bl_elapsed = time.perf_counter() - bl_start
        bl_total += bl_elapsed
        if cert:
            bl_certified += 1
        bl_str = "SAFE" if cert else "UNK"

        ab_str = "N/A"
        ab_elapsed = 0.0
        if abcrown_script is not None and abcrown_config is not None:
            spec_path = tmp_dir / f"sample_{i}.vnnlib"
            write_vnnlib(spec_path, center, predicted, NUM_CLASSES, args.eps)
            result, ab_elapsed = abcrown_verify(
                abcrown_script, onnx_path, spec_path,
                config_path=abcrown_config,
                input_shape=INPUT_SHAPE,
                timeout=args.abcrown_timeout,
            )
            ab_total += ab_elapsed
            ab_str = result.upper()
            if result.startswith("safe") or result in {"unsat", "holds"}:
                ab_safe += 1

        print(
            f"{f'sample_{i}':<10} {predicted:>4} {bl_str:>10} "
            f"{margin:>+10.4f} {ab_str:>10} "
            f"{bl_elapsed:>8.2f} {ab_elapsed:>8.2f}"
        )

    print("-" * len(header))
    n = args.n_samples
    bl_pct = 100 * bl_certified / n if n else 0.0
    ab_pct = 100 * ab_safe / n if n else 0.0
    summary = (
        f"{'Total':<10} {'':>4} "
        f"{f'{bl_certified}/{n} ({bl_pct:.0f}%)':>10} {'':>10} "
    )
    if abcrown_script is not None:
        summary += f"{f'{ab_safe}/{n} ({ab_pct:.0f}%)':>10} "
    else:
        summary += f"{'N/A':>10} "
    summary += f"{bl_total:>8.2f} {ab_total:>8.2f}"
    print(summary)


if __name__ == "__main__":
    main()
