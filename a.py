"""Concrete example where the new ``tanh_linearizer`` is looser than the original.

Both linearizers return an affine band

    slope * x + mu  +/-  beta   covering tanh(x) for x in [lb, ub].

The width of the band at any point is ``2 * beta``, so a larger ``beta`` means
a looser relaxation. This script picks an interval (found by random search)
where the new implementation reports a strictly larger ``beta``.
"""

import torch

from boundlab.zono.tanh import tanh_linearizer, tanh_linearizer_original


def report(name, bounds):
    slope = bounds.input_weights[0].item()
    mu = bounds.bias.item()
    beta = bounds.error_coeffs.tensor.item()
    print(f"  {name}: slope={slope:.10f}  mu={mu:.10f}  beta={beta:.6e}")
    return slope, mu, beta


# A small interval in the saturation regime — the original's analytic chord
# bound is essentially exact here, while the new softmax2-based bound carries
# a bit of slack.
lb = torch.tensor([2.630422])
ub = torch.tensor([2.640749])

print(f"interval: [{lb.item():.6f}, {ub.item():.6f}]   width={(ub-lb).item():.6e}")

orig = tanh_linearizer_original(ub, lb)
new = tanh_linearizer(ub, lb)

print()
print("affine bands:")
_, _, beta_o = report("original", orig)
_, _, beta_n = report("new     ", new)

print()
print(f"beta_new / beta_orig = {beta_n / beta_o:.4f}  (>1 means new is looser)")

# Soundness sanity check: sample many x in [lb, ub] and confirm tanh(x) sits
# inside both bands.
x = lb + (ub - lb) * torch.linspace(0, 1, 1001)
for name, b in [("original", orig), ("new", new)]:
    s = b.input_weights[0]
    m = b.bias
    e = b.error_coeffs.tensor
    upper = s * x + m + e
    lower = s * x + m - e
    f = torch.tanh(x)
    tol = 1e-6  # near float32 epsilon at these magnitudes
    assert (f <= upper + tol).all() and (f >= lower - tol).all(), f"{name} unsound"
print("both bands are sound on a 1001-point grid.")
