# Example: Export a BoundLab VNN Pipeline to ONNX

This example shows how to export a verification pipeline to ONNX.

The pipeline follows:

1. Build a neural network.
2. Apply `zono.interpret` to operate on symbolic expressions.
3. Concretize with `ublb()`.
4. Export the resulting `(ub, lb)` computation graph to ONNX.

## Script

Use:

```bash
python examples/export_vnn_to_onnx.py --output ./boundlab_vnn.onnx
```

Optional parameters:

```bash
python examples/export_vnn_to_onnx.py \
  --input-dim 64 \
  --width 128 \
  --depth 3 \
  --output-dim 32 \
  --eps-scale 0.1 \
  --output ./artifacts/vnn_bounds.onnx
```

## What Gets Exported

The exported ONNX model:

- input: concrete tensor `x`
- outputs: `ub`, `lb`

Internally, it encodes the BoundLab VNN process:

- symbolic abstract input `ConstVal(x) + eps`
- zonotope interpretation via `zono.interpret(...)`
- concretization via `ublb()`

## Source

See the full script at:

- `examples/export_vnn_to_onnx.py`
