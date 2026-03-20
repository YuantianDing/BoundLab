# Example: Transformer-Style Attention Block

This example shows how to analyze a small attention-like module using `zono.interpret`.

```python
import math
import torch
from torch import nn

import boundlab.expr as expr
import boundlab.zono as zono

class TinyAttention(nn.Module):
    def __init__(self, d_model=4, d_k=4):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        self.scale = math.sqrt(d_k)

    def forward(self, x):
        # x: [seq_len, d_model]
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        scores = torch.matmul(q, k.transpose(0, 1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


torch.manual_seed(0)
model = TinyAttention(d_model=3, d_k=3).eval()

seq_len, d_model = 2, 3
x_center = torch.randn(seq_len, d_model) * 0.2
x = expr.ConstVal(x_center) + 0.05 * expr.LpEpsilon([seq_len, d_model])

op = zono.interpret(model)
y = op(x)
ub, lb = y.ublb()

print("bounds ready")
print("max width:", (ub - lb).max().item())
```

## Notes

- The current softmax handler supports 2D tensors along the last dimension.
- Small perturbation radii are usually more stable for softmax-heavy examples.
- You can validate soundness by sampling concrete inputs and checking `lb <= output <= ub`.
