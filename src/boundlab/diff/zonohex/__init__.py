"""Hexagonal zonotopes for differetial verification.
"""

from typing import Literal, Union

import torch

from boundlab.diff import zono3
from boundlab.diff.zono3 import DiffZonoBounds
from boundlab.diff.expr import DiffExpr2, DiffExpr3
from boundlab.expr import TupleExpr
from boundlab.expr._core import Expr, ExprFlags
from boundlab.expr._tuple import GetTupleItem
from boundlab.interp import Interpreter
from boundlab.linearop._base import LinearOp
from boundlab.linearop._einsum import EinsumOp

class ZonoHexGate(TupleExpr):
    """Hexagonal Bound Propagation for differential verification.

    3 Inputs:
    - ``x``: The zonotope expression for the first network.
    - ``y``: The zonotope expression for the second network.
    - ``diff``: The zonotope expression for the difference between the two networks.

    2 Outputs:
    - ``x_bounded``
    - ``y_bounded``
    """
    def __init__(self, x: Expr, y: Expr, diff: Expr):
        assert x.shape == y.shape == diff.shape, f"Shapes of x, y, and diff must match; got {x.shape}, {y.shape}, {diff.shape}"
        out_flags = ExprFlags.NONE
        if all(bool(e.flags & ExprFlags.SYMMETRIC_TO_0) for e in (x, y, diff)):
            out_flags = ExprFlags.SYMMETRIC_TO_0
        # TupleExpr.__init__ takes one flag per output; ZonoHexGate has 2 outputs
        # (x_bounded, y_bounded) plus a dummy 3rd slot so that len(flags) matches
        # len(children) (the prop code uses len(children) as the output count).
        super().__init__(out_flags, out_flags, out_flags)
        self.x = x
        self.y = y
        self.diff = diff

    @property
    def shape(self) -> tuple[torch.Size, ...]:
        return (self.x.shape, self.y.shape)

    @property
    def children(self) -> tuple[Expr, ...]:
        """Children expressions that contribute to this TupleExpr. This is used for topological sorting and weight propagation."""
        return (self.x, self.y, self.diff)

    def backward(self, *weights: LinearOp, direction = "==") -> tuple[Union[torch.Tensor, Literal[0]], list] | None:
        """Perform backward-mode bound propagation through this expression."""
        # The prop code always passes len(children) == 3 weights; only the
        # first two correspond to real outputs. The 3rd is always 0 (dummy
        # output slot). Either x- or y-side weight may also be 0 when only
        # one of the two outputs is consumed downstream.
        x_weight = weights[0]
        y_weight = weights[1]

        # Determine the downstream ("output") shape from whichever weight is
        # non-zero, so we can fill the zero slot with matching shape.
        ref = x_weight if not (isinstance(x_weight, int) and x_weight == 0) else y_weight
        assert not (isinstance(ref, int) and ref == 0), \
            "ZonoHexGate.backward: at least one of x-/y-weight must be non-zero"
        downstream_shape = ref.output_shape

        if isinstance(x_weight, int) and x_weight == 0:
            return 0, [0, y_weight, 0]
        if isinstance(y_weight, int) and y_weight == 0:
            return 0, [x_weight, 0, 0]

        x_weight : EinsumOp =  x_weight.einsum_op()
        y_weight : EinsumOp = y_weight.einsum_op()
        x_weight = x_weight.permute_for_output()
        y_weight = y_weight.permute_for_output()
        if x_weight.input_dims == y_weight.input_dims and x_weight.output_dims == y_weight.output_dims:
            # If the weights have the same input and output dimensions, we can directly apply the zono-hex transformation to them.
            new_x_weight, new_y_weight, new_diff_weight = zono_hex_gate(x_weight.tensor, y_weight.tensor)
            new_x_weight = EinsumOp(new_x_weight, x_weight.input_dims, x_weight.output_dims)
            new_y_weight = EinsumOp(new_y_weight, y_weight.input_dims, y_weight.output_dims)
            new_diff_weight = EinsumOp(new_diff_weight, x_weight.input_dims, x_weight.output_dims)
            return 0, [new_x_weight, new_y_weight, new_diff_weight]
        else:
            x_weight = x_weight.jacobian()
            y_weight = y_weight.jacobian()
    
            new_x_weight, new_y_weight, new_diff_weight = zono_hex_gate(x_weight, y_weight)
    
            new_x_weight = EinsumOp.from_full(new_x_weight, len(self.x.shape))
            new_y_weight = EinsumOp.from_full(new_y_weight, len(self.y.shape))
            new_diff_weight = EinsumOp.from_full(new_diff_weight, len(self.diff.shape))
    
            return (0, [new_x_weight, new_y_weight, new_diff_weight])

    def with_children(self, *new_children: Expr) -> "TupleExpr":
        """Return a new TupleExpr with the same flags but new children. This is used for expression rewriting during bound propagation."""
        assert len(new_children) == 3, f"Expected 3 children for ZonoHexGate, got {len(new_children)}"
        return ZonoHexGate(*new_children)
    
    def split_const(self):
        """ZonoHexGate is a pure affine op, so the "const" part is just the bias."""
        return 0, self
    
    def to_string(self, *children_str: str, indent: int = 0) -> str:
        return f"<zonohex x={children_str[0]}, y={children_str[1]}, diff={children_str[2]}>"

    def simplify(self) -> "ZonoHexGate":
        from boundlab.prop import eqprop
        x_res = eqprop(self.x)
        assert all(not isinstance(e, ZonoHexGate) for e in x_res.all_subnodes()), \
            "ZonoHexGate.simplify: expected no ZonoHexGate nodes in simplified x expression"
        return ZonoHexGate(
            x_res,
            eqprop(self.y),
            eqprop(self.diff),
        )

def zono_hex_gate(x_weight: torch.Tensor, y_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cond_xy = (x_weight > 0) == (y_weight > 0) 
    cond_y_diff = (x_weight + y_weight >= 0) == (x_weight <= 0)
    cond_x_diff = ~(cond_xy | cond_y_diff)
    new_x_weight = torch.zeros_like(x_weight)
    new_x_weight = torch.where(cond_xy, x_weight, new_x_weight)
    new_x_weight = torch.where(cond_x_diff, x_weight + y_weight, new_x_weight)

    new_y_weight = torch.zeros_like(y_weight)
    new_y_weight = torch.where(cond_xy, y_weight, new_y_weight)
    new_y_weight = torch.where(cond_y_diff, x_weight + y_weight, new_y_weight)

    new_diff_weight = torch.zeros_like(x_weight)
    new_diff_weight = torch.where(cond_x_diff, -y_weight, new_diff_weight)
    new_diff_weight = torch.where(cond_y_diff, x_weight, new_diff_weight)
    return new_x_weight, new_y_weight, new_diff_weight

def expr3_to_expr2(expr3, **_kwargs) -> DiffExpr2:
    """Convert a DiffExpr3 to a DiffExpr2 via a :class:`ZonoHexGate` relaxation.

    Leaves non-:class:`DiffExpr3` inputs untouched so it can be chained after
    handlers that produce scalars, plain tensors, or :class:`DiffExpr2`.
    Extra keyword arguments are ignored (they come from the upstream ONNX
    handler's attributes via :class:`~boundlab.interp.FnListChain`).
    """
    if isinstance(expr3, DiffExpr3):
        Xc, Xs = expr3.x.split_const()
        Yc, Ys = expr3.y.split_const()
        gate = ZonoHexGate(Xs, Ys, expr3.diff - (Xc - Yc))
        return DiffExpr2(Xc + gate[0], Yc + gate[1])
    return expr3

interpret: Interpreter[Union[Expr, DiffExpr2]] = zono3.interpret.and_then(expr3_to_expr2)
