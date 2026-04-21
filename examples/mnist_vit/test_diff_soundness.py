"""Soundness test for certify_pruned_diff.py differential bounds."""
import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import torch
from torch import nn
import boundlab.expr as expr, boundlab.prop as prop, boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export
from mnist_vit import build_mnist_vit
from certify import PatchifyStage
from certify_pruned import ScoringModel, build_zonotope_no_cat, classify_topk
from certify_pruned_diff import MaskedModel, certify_int_sub, certify_zono_sub, certify_differential

vit = build_mnist_vit('mnist_transformer.pt')
eps = 0.004; K = 8

patchify = PatchifyStage(vit, True, 0.1307, 0.3081).eval()
gm_p = onnx_export(patchify, ([1, 28, 28],)); op_p = zono.interpret(gm_p)
scoring = ScoringModel(vit).eval()
gm_score = onnx_export(scoring, ([17, 64],)); op_score = zono.interpret(gm_score)

results = {n: [0, 0] for n in ['Int-Sub', 'Zono-Sub', 'Differential']}

for seed in range(5):
    torch.manual_seed(seed)
    img = (torch.rand(1, 28, 28) > 0.85).float()
    with torch.no_grad():
        x = (img - 0.1307) / 0.3081
        center = torch.cat((vit.cls_token[0], vit.to_patch_embedding(x)), dim=0) + vit.pos_embedding[0]

    # Classify tokens using zonotope bounds
    full_zono = build_zonotope_no_cat(vit, img, eps, op_p)
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    ub_sc, lb_sc = op_score(full_zono).ublb()
    definite_keep, definite_prune, uncertain = classify_topk(ub_sc, lb_sc, K)

    K_remaining = K - len(definite_keep)
    if K_remaining < 0: K_remaining = 0; uncertain = set()
    if K_remaining > len(uncertain): K_remaining = len(uncertain)
    uncertain_list = sorted(uncertain)

    from itertools import combinations
    if len(uncertain_list) == 0 or K_remaining == len(uncertain_list):
        cases = [definite_keep | uncertain]
    elif K_remaining == 0:
        cases = [definite_keep.copy()]
    else:
        cases = [definite_keep | set(c) for c in combinations(uncertain_list, K_remaining)]

    # Full model graph
    mask_full = torch.ones(17, 64)
    gm_full = onnx_export(MaskedModel(vit, mask_full).eval(), ([17, 64],))

    # Compute bounds with case splitting for each method
    bounds = {}
    for name, fn in [('Int-Sub', certify_int_sub), ('Zono-Sub', certify_zono_sub),
                      ('Differential', certify_differential)]:
        best_ub = best_lb = None
        for case_kept in cases:
            mp = torch.zeros(17, 64); mp[0] = 1.0
            for p in case_kept: mp[p + 1] = 1.0
            gm_pruned = onnx_export(MaskedModel(vit, mp).eval(), ([17, 64],))
            d_ub, d_lb = fn(gm_full, gm_pruned, center, eps)
            if best_ub is None:
                best_ub, best_lb = d_ub.clone(), d_lb.clone()
            else:
                best_ub = torch.maximum(best_ub, d_ub)
                best_lb = torch.minimum(best_lb, d_lb)
        bounds[name] = (best_ub, best_lb)

    # Test with true pruning on perturbed inputs
    model_full = MaskedModel(vit, mask_full).eval()
    for t in range(200):
        torch.manual_seed(seed * 10000 + t)
        delta = (2 * torch.rand(17, 64) - 1) * eps
        with torch.no_grad():
            xp = center + delta
            sc = scoring(xp)
            _, topk = sc.topk(K)
            kept = set(topk.tolist())
            mp = torch.zeros(17, 64); mp[0] = 1.0
            for p in kept: mp[p + 1] = 1.0
            model_pruned = MaskedModel(vit, mp).eval()
            diff = model_full(xp) - model_pruned(xp)
        for name, (d_ub, d_lb) in bounds.items():
            ok = ((diff >= d_lb - 1e-4) & (diff <= d_ub + 1e-4)).all().item()
            results[name][0] += int(ok)
            results[name][1] += 1
            if not ok:
                viol = max((d_lb - diff).clamp(min=0).max().item(),
                           (diff - d_ub).clamp(min=0).max().item())
                print(f'FAIL {name} seed={seed} t={t} viol={viol:.6f}')

    print(f'seed {seed} done (cases={len(cases)}, unc={len(uncertain)})')

print()
for name, (p, tot) in results.items():
    status = "SOUND" if p == tot else "UNSOUND"
    print(f'{name:<15} {p}/{tot} {status}')