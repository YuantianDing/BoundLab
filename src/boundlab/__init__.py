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

from importlib import import_module

import torch

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "linearop",
    "expr",
    "prop",
    "interp",
    "zono",
    "poly",
    "diff",
    "utils",
    "gradlin"
]

def __getattr__(name: str):
    if name in __all__ and name != "__version__":
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

torch.backends.opt_einsum.enabled = True
torch.backends.opt_einsum.strategy = "optimal"
