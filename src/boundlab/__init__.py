"""BoundLab - A framework for building neural network verification tools.

Examples
--------
Create a simple zonotope input and concretize bounds:

>>> import torch
>>> import boundlab.expr as expr
>>> x = expr.ConstVal(torch.tensor([0.0, 1.0])) + expr.LpEpsilon([2])
>>> x.ublb()[0].shape
torch.Size([2])
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "linearop"
    "expr",
    "prop",
    "interp",
    "zono",
    "poly",
    "diff",
    "utils",
]

from boundlab import expr, prop, zono, utils, poly, interp, diff
