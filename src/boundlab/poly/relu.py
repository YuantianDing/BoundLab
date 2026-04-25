"""ReLU linearizer for polytope abstract interpretation.

CROWN-style relaxation of :math:`\\mathrm{ReLU}(x) = \\max(0, x)`.
"""

import torch

from . import PolyBounds, _register_linearizer


@_register_linearizer("relu")
def relu_linearizer(ub: torch.Tensor, lb: torch.Tensor) -> PolyBounds:
    r"""CROWN relaxation of ReLU.

    For each neuron with input bounds :math:`[\ell, u]`:

    - Dead (:math:`u \le 0`): ``f(x) = 0``.
    - Active (:math:`\ell \ge 0`): ``f(x) = x``.
    - Crossing (:math:`\ell < 0 < u`): tight upper envelope through
      :math:`(\ell, 0)` and :math:`(u, u)`; lower slope chosen as 1 when
      :math:`u \ge -\ell` and 0 otherwise (the standard CROWN-lb rule).

    Examples
    --------
    >>> import torch
    >>> from boundlab.poly.relu import relu_linearizer
    >>> ub = torch.tensor([2.0, -1.0, 1.0])
    >>> lb = torch.tensor([-1.0, -2.0, 0.5])
    >>> b = relu_linearizer(ub, lb)
    >>> b.upper_lam.shape
    torch.Size([3])
    """
    zero = torch.zeros_like(ub)
    one = torch.ones_like(ub)

    active = lb >= 0
    dead = ub <= 0
    crossing = ~(active | dead)

    cross_slope = ub / (ub - lb + 1e-30)
    cross_upper_bias = -cross_slope * lb
    cross_lower_slope = torch.where(ub >= -lb, one, zero)

    upper_lam = torch.where(active, one, torch.where(crossing, cross_slope, zero))
    upper_bias = torch.where(crossing, cross_upper_bias, zero)
    lower_lam = torch.where(active, one, torch.where(crossing, cross_lower_slope, zero))
    lower_bias = torch.zeros_like(lb)

    return PolyBounds(
        upper_lam=upper_lam,
        upper_bias=upper_bias,
        lower_lam=lower_lam,
        lower_bias=lower_bias,
    )
