"""Smoke test: verify certify_differential_split runs end-to-end on a
tiny synthetic model and gives tighter bounds than budget=1."""

import sys
import torch
from torch import nn

# Stub: tiny two-layer FFN with tanh. Easy to check against ground truth.
class TinyNet(nn.Module):
    def __init__(self, d_in=4, d_hidden=8, d_out=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


def main():
    # Two models, slightly perturbed
    m1 = TinyNet(seed=0).eval()
    m2 = TinyNet(seed=0).eval()
    # Perturb m2's weights slightly (like a quantization effect)
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(0.02 * torch.randn_like(p))

    center = torch.zeros(4)
    eps = 0.05

    # Import here to surface import errors clearly
    from boundlab.diff.certify_split import certify_differential_split

    print("Running budget=1 (no split)...")
    ub1, lb1 = certify_differential_split(m1, m2, center, eps, budget=1, verbose=True)
    w1 = (ub1 - lb1).max().item()
    print(f"  budget=1 max width: {w1:.6f}\n")

    print("Running budget=8...")
    ub8, lb8 = certify_differential_split(m1, m2, center, eps, budget=8, verbose=True)
    w8 = (ub8 - lb8).max().item()
    print(f"  budget=8 max width: {w8:.6f}\n")

    print("Running budget=32...")
    ub32, lb32 = certify_differential_split(m1, m2, center, eps, budget=32, verbose=True)
    w32 = (ub32 - lb32).max().item()
    print(f"  budget=32 max width: {w32:.6f}\n")

    # Soundness vs Monte Carlo
    torch.manual_seed(1)
    n_mc = 5000
    samples = center.unsqueeze(0) + (torch.rand(n_mc, 4) * 2 - 1) * eps
    with torch.no_grad():
        mc_diffs = torch.stack([m1(s) - m2(s) for s in samples])
    mc_ub = mc_diffs.max(0).values
    mc_lb = mc_diffs.min(0).values
    mc_width = (mc_ub - mc_lb).max().item()
    print(f"  MC width (n={n_mc}): {mc_width:.6f}")

    # Soundness check
    def is_sound(ub, lb, mc_ub, mc_lb, tol=1e-6):
        return (ub >= mc_ub - tol).all().item() and (lb <= mc_lb + tol).all().item()
    print(f"\nSoundness:")
    print(f"  budget=1:  {is_sound(ub1, lb1, mc_ub, mc_lb)}")
    print(f"  budget=8:  {is_sound(ub8, lb8, mc_ub, mc_lb)}")
    print(f"  budget=32: {is_sound(ub32, lb32, mc_ub, mc_lb)}")

    print(f"\nTightness (smaller = better):")
    print(f"  budget=1:  {w1:.6f}")
    print(f"  budget=8:  {w8:.6f}  (ratio {w1/w8:.2f}x)")
    print(f"  budget=32: {w32:.6f}  (ratio {w1/w32:.2f}x)")
    print(f"  MC floor:  {mc_width:.6f}")

    ok = (w32 <= w1 + 1e-8) and is_sound(ub32, lb32, mc_ub, mc_lb)
    print(f"\n{'PASS' if ok else 'FAIL'}: split monotonically tightens and is sound")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
