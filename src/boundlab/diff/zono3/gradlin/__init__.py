"""Gradient-descent-tightened differential linearizers.

Each linearizer here bounds ``f(x) − f(y)`` over the
per-neuron trapezoid ``{(x,y) : lx≤x≤ux, ly≤y≤uy, ld≤x-y≤ud}`` by running
:func:`boundlab.gradlin.gradlin` — an Adam-based optimiser over the linear
slopes. Compared with the closed-form defaults in
:mod:`boundlab.diff.zono3.default`, these are slower per call but often
tighter, especially on asymmetric ``x``/``y`` ranges where ``lam_x`` and
``lam_y`` should differ.

To use them, register against the live interpreter::

    from boundlab.diff.zono3 import interpret, linearizer_to_hander
    from boundlab.diff.zono3 import gradlin as gl
    gl.register_all(interpret, linearizer_to_hander)
"""

from __future__ import annotations

from ._common import make_unary_diff_linearizer
from .exp import exp_linearizer
from .tanh import tanh_linearizer
from .reciprocal import reciprocal_linearizer
from .relu import relu_linearizer


def register_all(interpret, linearizer_to_hander) -> None:
    """Install gradlin linearizers onto *interpret* in place.

    Replaces/augments ``Relu``, ``Exp``, ``Tanh``, ``Reciprocal`` (plus
    lowercase ATen aliases) with gradlin-backed handlers. Other operators
    (``Mul``, ``Softmax``, …) are left on their default implementations.
    """
    for linearizer, names in [
        # (relu_linearizer, ("relu", "Relu")),
        (exp_linearizer, ("exp", "Exp")),
        (tanh_linearizer, ("tanh", "Tanh")),
        (reciprocal_linearizer, ("reciprocal", "Reciprocal")),
    ]:
        handler = linearizer_to_hander(linearizer)
        for name in names:
            interpret[name] = handler
    from boundlab.diff.zono3.default.softmax import diff_softmax_handler
    
    interpret["Softmax"] = lambda X, axis=-1: diff_softmax_handler(X, dim=axis, exp_handler=interpret["Exp"], reciprocal_handler=interpret["Reciprocal"])

__all__ = [
    "make_unary_diff_linearizer",
    "relu_linearizer",
    "exp_linearizer",
    "tanh_linearizer",
    "reciprocal_linearizer",
    "register_all",
]
