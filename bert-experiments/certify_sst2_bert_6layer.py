#!/usr/bin/env python3
"""Wrapper to run certify_sst2_bert on the 6-layer, 256-d checkpoint.

Checkpoint must live at BoundLab/model6/ with:
    config.json, vocab.txt, pytorch_model.bin

Usage:
    python3 experiments/certify_sst2_bert_6layer.py [any --args forwarded]

Defaults match the 6-layer checkpoint's architecture. You can override via
argparse flags as usual — e.g.:
    python3 experiments/certify_sst2_bert_6layer.py --eps 0.005 --bits 8 \\
                                                   --n-samples 1 --max-len 10

This is identical to running:
    BOUNDLAB_MODEL_DIR=model6 python3 experiments/certify_sst2_bert.py \\
        --layers 6 [your args]
but with the env var + --layers 6 pre-set so you don't have to remember.
"""

import os
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    certify_script = here / "certify_sst2_bert.py"

    if not certify_script.exists():
        print(f"ERROR: {certify_script} not found", file=sys.stderr)
        sys.exit(1)

    root = here.parent
    model6_dir = root / "model6"
    if not model6_dir.exists():
        print(f"ERROR: expected checkpoint directory at {model6_dir}", file=sys.stderr)
        print("  Create it and populate with config.json, vocab.txt, pytorch_model.bin",
              file=sys.stderr)
        sys.exit(1)

    # Force MODEL_DIR to model6 via env var (requires the A1 edit to
    # certify_sst2_bert.py).
    os.environ["BOUNDLAB_MODEL_DIR"] = "model6"

    # Forward all user args, but inject --layers 6 if the user didn't override.
    user_args = sys.argv[1:]
    if not any(a == "--layers" or a.startswith("--layers=") for a in user_args):
        user_args = ["--layers", "6"] + user_args

    # Replace this process with the real script so argparse, signals, etc.
    # behave identically to invoking it directly.
    argv = [sys.executable, str(certify_script)] + user_args
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
