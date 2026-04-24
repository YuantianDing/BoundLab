"""Default differential zonotope linearizers.

These are the closed-form linearizers derived analytically (VeryDiff-style
for ReLU; mean-derivative relaxations for exp/tanh/reciprocal). An
alternative group of gradient-descent-tightened linearizers lives in
:mod:`boundlab.diff.zono3.gradlin`.
"""

from .relu import relu_linearizer
from .tanh import tanh_linearizer
from .exp import exp_linearizer
from .reciprocal import reciprocal_linearizer
from .bilinear import (
    diff_mul_handler,
    diff_matmul_handler,
    diff_bilinear_elementwise,
    diff_bilinear_matmul,
)
from .softmax import diff_softmax_handler
from .heaviside import diff_heaviside_pruning_handler

__all__ = [
    "relu_linearizer",
    "tanh_linearizer",
    "exp_linearizer",
    "reciprocal_linearizer",
    "diff_mul_handler",
    "diff_matmul_handler",
    "diff_bilinear_elementwise",
    "diff_bilinear_matmul",
    "diff_softmax_handler",
    "diff_heaviside_pruning_handler",
]
