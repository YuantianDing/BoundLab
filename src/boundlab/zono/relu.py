

from boundlab.expr._core import Expr

from . import ZonoBounds, _register_linearizer

@_register_linearizer("relu")
def relu_linearizer(*expr: Expr) -> ZonoBounds:
    # TODO: Implement the actual linearization logic for ReLU expressions, using the input bounds to compute the output zonotope bounds.
    pass