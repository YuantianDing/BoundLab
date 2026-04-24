"""DeepT zonotope certification for the CIFAR-10 ViT at examples/vit/vit.py.

Adapts the DeepT zonotope propagation (compare/deept/Zonotope.py) to the
architecture of examples/vit/vit.py, which differs from the original DeepT
MNIST ViT in patch embedding (Conv2d vs linear), pooling (mean vs CLS-token)
and module naming.

Usage (from the boundlab root):
    pixi run python compare/deept/certify_cifar_vit.py
    pixi run python compare/deept/certify_cifar_vit.py --model pgd_2_3_16 --eps 0.001 --n-samples 3 --p 2
"""
from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))                          # local Zonotope.py
sys.path.insert(0, str(ROOT / "examples"))             # examples/vit/vit.py

from Zonotope import Zonotope, make_zonotope_new_weights_same_args  # type: ignore
from vit import vit as vit_module


def _boxify_4d(z: "Zonotope", lb_clamp: float | None = None, ub_clamp: float | None = None) -> "Zonotope":
    """Replace a (A, 1+E, N, N) zonotope with an independent-per-element box abstraction.

    Concretize the current bounds, optionally clamp them into [lb_clamp, ub_clamp], and
    build a fresh zonotope whose weights are finite by construction: one independent
    error symbol per (A, i, j) position carrying the radius. Used after softmax when
    the propagated bounds have saturated our numerical clamps and the zonotope_w tensor
    holds +inf / very-large entries that would turn into NaN under downstream matmul.
    """
    lb, ub = z.concretize()
    # Replace NaN/inf bounds with the clamp limits so we still produce a finite box.
    lb_fill = lb_clamp if lb_clamp is not None else -1e6
    ub_fill = ub_clamp if ub_clamp is not None else 1e6
    lb = torch.nan_to_num(lb, nan=lb_fill, posinf=ub_fill, neginf=lb_fill)
    ub = torch.nan_to_num(ub, nan=ub_fill, posinf=ub_fill, neginf=lb_fill)
    if lb_clamp is not None:
        lb = lb.clamp(min=lb_clamp)
    if ub_clamp is not None:
        ub = ub.clamp(max=ub_clamp)
    lb = torch.minimum(lb, ub)  # in case clamping crossed the bounds
    center = 0.5 * (lb + ub)
    radius = 0.5 * (ub - lb)

    A, N1, N2 = center.shape
    num_new = A * N1 * N2
    w = torch.zeros(A, 1 + num_new, N1, N2, device=center.device, dtype=center.dtype)
    w[:, 0] = center
    flat_radius = radius.reshape(-1)
    a_idx = torch.arange(A).repeat_interleave(N1 * N2)
    i_idx = torch.arange(N1).repeat_interleave(N2).repeat(A)
    j_idx = torch.arange(N2).repeat(A * N1)
    err_idx = torch.arange(num_new) + 1
    w[a_idx, err_idx, i_idx, j_idx] = flat_radius
    return make_zonotope_new_weights_same_args(w, source_zonotope=z, clone=False)


# ---------------------------------------------------------------------------
# DeepT args Namespace
# ---------------------------------------------------------------------------

def _deept_args(num_pixels: int, p: float = float("inf"), device: str = "cpu") -> Namespace:
    return Namespace(
        perturbed_words=1,
        attack_type="l_inf",
        all_words=True,
        device=device,
        cpu=(device == "cpu"),
        num_input_error_terms=num_pixels,
        zonotope_slow=False,
        error_reduction_method="box",
        p=p,
        add_softmax_sum_constraint=True,
        use_dot_product_variant3=False,
        use_other_dot_product_ordering=False,
        num_perturbed_words=1,
        concretize_special_norm_error_together=True,
        batch_softmax_computation=False,
        keep_intermediate_zonotopes=False,
        max_num_error_terms=14_000,
    )


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def certify_sample(
    model: vit_module.ViT,
    image: torch.Tensor,
    eps: float,
    p: float = float("inf"),
    max_error_terms: int = 14_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propagate DeepT zonotope bounds through the CIFAR-10 ViT.

    Returns (lb, ub) tensors of shape (num_classes,).
    """
    patch_size = model.patch_conv.kernel_size[0]
    C, H, W = image.shape
    num_patches = (H // patch_size) * (W // patch_size)
    patch_dim = C * patch_size * patch_size

    args = _deept_args(num_pixels=image.numel(), p=p)

    # 1. Extract patches: (C, H, W) → (num_patches, patch_dim)
    patches = (image
               .reshape(C, H // patch_size, patch_size, W // patch_size, patch_size)
               .permute(1, 3, 0, 2, 4)
               .reshape(num_patches, patch_dim))

    # 2. Build zonotope over the patch pixels (eps per pixel, all_words=True)
    z = Zonotope(args, p=p, eps=eps, perturbed_word_index=None,
                 value=patches, start_perturbation=0, end_perturbation=num_patches)

    # 3. Patch embedding: Conv2d(kernel=stride=p) ≡ Linear on each patch
    conv = model.patch_conv
    out_c, in_c, kH, kW = conv.weight.shape
    import torch.nn as nn
    patch_linear = nn.Linear(patch_dim, out_c, bias=conv.bias is not None)
    with torch.no_grad():
        patch_linear.weight.copy_(conv.weight.detach().reshape(out_c, -1))
        if conv.bias is not None:
            patch_linear.bias.copy_(conv.bias.detach())
    z = z.dense(patch_linear)               # (1+E, num_patches, dim)

    # 4. Prepend CLS token (constant — zero error coefficients)
    dim = model.cls_token.shape[0]
    E1 = z.zonotope_w.shape[0]
    cls_w = torch.zeros(E1, 1, dim)
    cls_w[0, 0] = model.cls_token.detach()
    full_w = torch.cat([cls_w, z.zonotope_w], dim=1)   # (1+E, 1+num_patches, dim)
    z = make_zonotope_new_weights_same_args(full_w, source_zonotope=z, clone=False)

    # 5. Add position embeddings (constant)
    pos = model.pos_embedding.detach()[:1 + num_patches]
    z = z.add(pos)

    # 6. Transformer blocks
    for block in model.blocks:
        if z.num_error_terms > max_error_terms:
            z = z.reduce_num_error_terms_box(max_num_error_terms=max_error_terms)

        # Attention
        z_ln = z.layer_norm(block.attn_norm, model.layer_norm_type)
        # Reduce error terms before the bilinear dot product to keep exp bounds finite.
        z_ln = z_ln.reduce_num_error_terms_box(max_num_error_terms=max_error_terms)
        h = block.attn.heads
        q = z_ln.dense(block.attn.to_q).add_attention_heads_dim(h)
        k = z_ln.dense(block.attn.to_k).add_attention_heads_dim(h)
        v = z_ln.dense(block.attn.to_v).add_attention_heads_dim(h)

        scores = q.dot_product(k).multiply(block.attn.scale)
        # Numerically stable softmax: subtract a per-row concretized upper bound from
        # the score center. Softmax is invariant to per-row additive shifts, and this
        # keeps the exp() inputs' upper bound ≤ 0 so exp outputs stay in [0, 1], which
        # is crucial with IBP attention logits that can span ±400 (float32 exp would
        # overflow). The max along the key dim is applied to the center only.
        scores_l, scores_u = scores.concretize()
        shift = scores_u.max(dim=-1, keepdim=True).values.detach()
        scores_w = scores.zonotope_w.clone()
        scores_w[:, 0] = scores_w[:, 0] - shift
        scores = make_zonotope_new_weights_same_args(scores_w, source_zonotope=scores, clone=False)
        # use_new_reciprocal=False picks the simpler lambda=-1/u² branch, which is
        # numerically safer under the reciprocal clamp when IBP attention logits have
        # very wide bounds (the new-reciprocal branch can produce 0/0 in mean_slope).
        # use_new_softmax=False uses the non-diff path sum(exp(x))·reciprocal, which
        # benefits from the max-subtraction above. We skip the softmax-sum equality
        # constraint (no_constraints=True) because its Gauss-elimination step is not
        # numerically robust on our loose bounds.
        attn = scores.softmax(no_constraints=True,
                              use_new_softmax=False,
                              use_new_reciprocal=False)
        # Sanitize: softmax outputs live in [0, 1], but our clamped exp/reciprocal
        # can leave +inf entries in attn.zonotope_w that cascade to NaN in the next
        # bilinear matmul. Box-relax to [0, 1] — loose but sound w.r.t. softmax.
        attn = _boxify_4d(attn, lb_clamp=0.0, ub_clamp=1.0)
        ctx = attn.dot_product(v.t()).remove_attention_heads_dim()
        attn_out = ctx.dense(block.attn.to_out)

        z = attn_out.add(z.expand_error_terms_to_match_zonotope(attn_out))  # residual 1

        # FFN
        z_ln2 = z.layer_norm(block.ff_norm, model.layer_norm_type)
        ff = z_ln2.dense(block.ff.net[0]).relu().dense(block.ff.net[2])

        z = ff.add(z.expand_error_terms_to_match_zonotope(ff))              # residual 2

    # 7. Mean pooling: average over all (1+num_patches) tokens
    z_mean_w = z.zonotope_w.mean(dim=1, keepdim=True)
    z = make_zonotope_new_weights_same_args(z_mean_w, source_zonotope=z, clone=False)

    # 8. Head
    z = z.layer_norm(model.head_norm, model.layer_norm_type)
    z = z.dense(model.head_linear)

    lb, ub = z.concretize()
    return lb.detach().squeeze(), ub.detach().squeeze()


# ---------------------------------------------------------------------------
# Certification loop
# ---------------------------------------------------------------------------

def certify_vit(
    *,
    model_name: str = "ibp_3_3_8",
    layer_norm_type: str = "no_var",
    eps: float = 0.002,
    p: float = float("inf"),
    n_samples: int = 5,
    seed: int = 0,
    max_error_terms: int = 14_000,
) -> dict:
    torch.manual_seed(seed)

    ctor = {"ibp_3_3_8": vit_module.vit_ibp_3_3_8,
            "pgd_2_3_16": vit_module.vit_pgd_2_3_16}[model_name]

    model = ctor(layer_norm_type=layer_norm_type).eval()
    model.layer_norm_type = layer_norm_type   # expose for propagation code

    results = []
    for i in range(n_samples):
        center = torch.rand(3, 32, 32)
        with torch.no_grad():
            predicted = int(model(center).argmax().item())

        t0 = time.time()
        lb, ub = certify_sample(model, center, eps=eps, p=p,
                                max_error_terms=max_error_terms)
        elapsed = time.time() - t0

        ub_others = ub.clone()
        ub_others[predicted] = float("-inf")
        margin = float(lb[predicted] - ub_others.max())
        certified = margin > 0.0

        results.append(dict(sample=i+1, predicted=predicted,
                            certified=certified, margin=margin, elapsed=elapsed))
        tag = "CERTIFIED" if certified else "not certified"
        print(f"  [{i+1}/{n_samples}] class={predicted:2d}  {tag}  "
              f"margin={margin:+.4f}  ({elapsed:.1f}s)")

    n_certified = sum(r["certified"] for r in results)
    return {"n_certified": n_certified, "n_total": n_samples, "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepT ViT certification on CIFAR-10.")
    parser.add_argument("--model", choices=["ibp_3_3_8", "pgd_2_3_16"], default="ibp_3_3_8")
    parser.add_argument("--layer-norm", choices=["standard", "no_var"], default="no_var", dest="layer_norm")
    parser.add_argument("--eps", type=float, default=0.002)
    parser.add_argument("--p", type=float, default=float("inf"))
    parser.add_argument("--n-samples", type=int, default=5, dest="n_samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-error-terms", type=int, default=14_000, dest="max_error_terms")
    args = parser.parse_args()

    p_str = "inf" if args.p > 10 else str(int(args.p))
    print("=" * 58)
    print("  DeepT Certification — CIFAR-10 ViT (examples/vit/vit.py)")
    print("=" * 58)
    print(f"  model      : {args.model}")
    print(f"  layer_norm : {args.layer_norm}")
    print(f"  eps (L{p_str})  : {args.eps}")
    print(f"  n_samples  : {args.n_samples}")
    print(f"  seed       : {args.seed}")
    print("=" * 58)
    print()

    result = certify_vit(
        model_name=args.model, layer_norm_type=args.layer_norm,
        eps=args.eps, p=args.p, n_samples=args.n_samples,
        seed=args.seed, max_error_terms=args.max_error_terms,
    )
    n_c, n_t = result["n_certified"], result["n_total"]
    print(f"\nCertified {n_c}/{n_t} ({100*n_c/n_t if n_t else 0:.0f}%)")
