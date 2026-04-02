# Examples

These examples are organized from simple symbolic construction to full-model interpretation.

```{toctree}
:maxdepth: 1

manual_bounds
interpreter_mlp
softmax_attention
vnn_to_onnx
diff_verification
```

## Choosing an Example

- Start with {doc}`manual_bounds` to learn expression and bound APIs.
- Continue with {doc}`interpreter_mlp` to verify an exported MLP graph.
- {doc}`softmax_attention` covers transformer-style operations (`matmul`, `softmax`).
- {doc}`vnn_to_onnx` shows how to export a BoundLab verification pipeline to ONNX.
- {doc}`diff_verification` demonstrates certifying bounds on the *difference* between two networks.
