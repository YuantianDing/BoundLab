r"""Base Expression Classes

This module provides the ConstVal expression, a thin wrapper around
AffineSum(const=...) for constant tensor values.
"""

import torch

from boundlab.expr._linear import AffineSum


class ConstVal(AffineSum):
    """Expression representing a constant tensor value.

    Implemented as an AffineSum with no children and only a constant term.
    When used as a child of another AffineSum, the constant is automatically
    absorbed via eager contraction.
    """

    def __init__(self, value: torch.Tensor, name: str | None = None):
        super().__init__(const=value)
        self.value = value
        self.name = name

    def to_string(self) -> str:
        if self.name is not None:
            return f"#const {self.name}"
        return f"#const <{self.id:X}>"
