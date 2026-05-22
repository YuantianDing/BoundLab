"""Backwards-compatibility shim — maps old names to new modules.

Existing scripts that ``from pruning import ...`` continue to work.
Migrate to importing from ``pipeline``, ``token_pruning``, or
``masked_softmax`` directly, then this file can be deleted.
"""
import torch

# New modules — direct re-exports
from pipeline import ScoringModel, PrunedViT                    # noqa: F401
from token_pruning import (                                      # noqa: F401
    build_token_mask, build_full_token_mask,
    build_token_scores, build_all_kept_scores,
    classify_topk, enumerate_pruning_cases,
    build_input_zonotope, export_patch_embedding,
    export_scoring, export_pruned_vit,
    CertifyResult, certify_pruning_diff,
)
from masked_softmax import masked_softmax_op                     # noqa: F401
from boundlab.interp.onnx import onnx_export


# ---------------------------------------------------------------------------
# Old-API adapters (emb_mask → token_scores)
# ---------------------------------------------------------------------------

def _mask_to_scores(emb_mask: torch.Tensor, magnitude: float = 100.0) -> torch.Tensor:
    """Convert (N+1, D) binary mask → (N+1,) token scores."""
    col_mask = emb_mask[:, 0]
    return (col_mask * 2 - 1) * magnitude


def MaskedPostConcat(vit, emb_mask, mask_from_layer=0, use_custom_op=False):
    """Old API: wraps PrunedViT, converting emb_mask to token_scores."""
    scores = _mask_to_scores(emb_mask)
    return PrunedViT(vit, scores, mask_from_layer=mask_from_layer,
                     for_verification=use_custom_op)


def export_masked_post_concat(vit, emb_mask, num_tokens, dim, mask_from_layer=0):
    """Old API: wraps export_pruned_vit, converting emb_mask to kept_patches."""
    col_mask = emb_mask[:, 0]
    kept = {i for i in range(num_tokens) if col_mask[i + 1] > 0.5}
    return export_pruned_vit(vit, kept, num_tokens, dim, mask_from_layer)


# Old names → new names (compatible signatures)
build_emb_mask = build_token_mask
build_full_emb_mask = build_full_token_mask
build_zonotope_no_cat = build_input_zonotope
certify_pruned_sample_diff = certify_pruning_diff