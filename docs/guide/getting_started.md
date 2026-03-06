# Getting Started

## Installation

### From conda (recommended)

```bash
conda install boundlab -c pytorch -c conda-forge
```

### From source

```bash
git clone https://github.com/YuantianDing/boundlab.git
cd boundlab
pip install -e .
```

## Quick Start

```python
import torch
from torch.nn import functional as F
import boundlab as bl

class MyModel(nn.Module):
    def forward(self, x):
        return F.relu(x)

inputs = bl.expr.const(load_a_batch_of_data())
input_noise = 0.1 * bl.expr.var.LInfEpsilon(*inputs.shape)

zonotope = bl.zono.operator(MyModel())(inputs + input_noise)
ub, lb = zonotope.ublb()
assert (ub - lb).max() < 0.2  # Check that the output is robust to the input noise
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
