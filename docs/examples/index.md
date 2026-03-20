# Examples

These examples are organized from simple symbolic construction to full-model interpretation.

```{toctree}
:maxdepth: 1

manual_bounds
interpreter_mlp
softmax_attention
vnn_to_onnx
```

## Choosing an Example

- Start with {doc}`manual_bounds` to learn expression and bound APIs.
- Continue with {doc}`interpreter_mlp` to verify an `nn.Sequential` model.
- Use {doc}`softmax_attention` for transformer-style operations (`matmul`, `softmax`).
- Use {doc}`vnn_to_onnx` to export a BoundLab verification pipeline to ONNX.
