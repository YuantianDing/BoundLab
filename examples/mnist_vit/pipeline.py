"""Model wrappers for token-pruning verification.

``ScoringModel``
    Extracts per-patch importance scores from CLS attention.
    ``(N+1, D) → (N,)``

``PrunedViT``
    Runs the ViT transformer + classification head with a token pruning
    mask derived from ``token_scores``.
    ``(N+1, D) → (num_classes,)``
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

# Handler registrations — importing activates them
import masked_softmax as _msm          # noqa: F401  (registers MaskedSoftmax + Mul)
import heaviside_handler as _hh         # noqa: F401  (registers HeavisidePruning)

from boundlab.diff.op import heaviside_pruning


# ---------------------------------------------------------------------------
# ScoringModel
# ---------------------------------------------------------------------------

class ScoringModel(nn.Module):
    """Extract per-patch importance from CLS attention.

    Computes ``softmax(Q_cls @ K^T / √d)`` averaged over heads,
    returning one score per patch token (CLS excluded).

    Input:  ``(N+1, D)`` token embeddings (CLS + patches + pos_emb already added).
    Output: ``(N,)`` importance scores per patch.

    Parameters
    ----------
    vit : ViT
        Source model.
    score_layer : int
        Which transformer layer's CLS attention to use.  Layers
        ``0..score_layer-1`` are run first (unmasked).
    """

    def __init__(self, vit, score_layer: int = 0):
        super().__init__()
        assert 0 <= score_layer < len(vit.transformer.layers)
        self.score_layer = score_layer
        self.prefix_layers = nn.ModuleList()
        for i in range(score_layer):
            attn_block, ff_block = vit.transformer.layers[i]
            self.prefix_layers.append(nn.ModuleList([attn_block, ff_block]))
        attn_block = vit.transformer.layers[score_layer][0]
        self.norm = attn_block.fn.norm
        self.attn = attn_block.fn.fn
        self.heads = self.attn.heads
        self.dim_head = self.attn.dim_head
        self.scale = self.attn.scale

    def forward(self, x: Tensor) -> Tensor:
        for attn_block, ff_block in self.prefix_layers:
            x = attn_block(x)
            x = ff_block(x)
        xn = self.norm(x)
        n = xn.shape[0]
        h, d = self.heads, self.dim_head
        Q = self.attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)
        K = self.attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
        Q_cls = Q[:, 0:1, :]
        scores = (Q_cls @ K.transpose(-2, -1)) * self.scale
        attn_weights = scores.softmax(dim=-1)
        importance = attn_weights.mean(dim=0).squeeze(0)
        return importance[1:]


# ---------------------------------------------------------------------------
# PrunedViT
# ---------------------------------------------------------------------------

class PrunedViT(nn.Module):
    """ViT transformer + classification head with token pruning.

    Tokens are pruned based on ``token_scores``: positive scores are kept,
    negative scores are zeroed out.  Pruning is applied at
    ``mask_from_layer`` and affects all subsequent attention layers.

    Input:  ``(N+1, D)`` token embeddings (CLS + patches + pos_emb already added).
    Output: ``(num_classes,)`` logits.

    Parameters
    ----------
    vit : ViT
        Source model (any depth).
    token_scores : Tensor
        Shape ``(N+1,)``.  Positive → kept, negative → pruned.
        CLS (index 0) should always be positive.
        Use ``build_token_scores()`` from ``token_pruning.py``.
    mask_from_layer : int
        First transformer layer at which pruning is applied.
        Layers before this use standard softmax with no masking.
    for_verification : bool
        If True, uses ``heaviside_pruning`` ONNX ops (for export +
        abstract interpretation).  If False, uses concrete masking
        (for Monte Carlo sampling).
    """

    def __init__(self, vit, token_scores: Tensor, mask_from_layer: int = 0,
                 for_verification: bool = False):
        super().__init__()
        self.pool = vit.pool
        self.mlp_head = vit.mlp_head
        self.mask_from_layer = mask_from_layer
        self.for_verification = for_verification
        self.register_buffer("token_scores", token_scores)

        self.n_layers = len(vit.transformer.layers)
        self.attn_norms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.ff_blocks = nn.ModuleList()
        self.heads_list = []
        self.dim_head_list = []
        self.scale_list = []
        for attn_block, ff_block in vit.transformer.layers:
            self.attn_norms.append(attn_block.fn.norm)
            self.attns.append(attn_block.fn.fn)
            self.ff_blocks.append(ff_block)
            self.heads_list.append(attn_block.fn.fn.heads)
            self.dim_head_list.append(attn_block.fn.fn.dim_head)
            self.scale_list.append(attn_block.fn.fn.scale)

    def _apply_token_mask(self, x: Tensor) -> Tensor:
        """Zero out pruned tokens."""
        if self.for_verification:
            scores_2d = self.token_scores.unsqueeze(-1).expand_as(x)
            return heaviside_pruning(scores_2d, x)
        else:
            mask = (self.token_scores >= 0).float().unsqueeze(-1)
            return x * mask

    def _masked_attention_weights(self, raw_scores: Tensor, n: int) -> Tensor:
        """Compute masked softmax attention weights.

        Decomposes masked softmax as: pairwise diff → exp → mask → sum → reciprocal.

        For verification: mask step uses ``heaviside_pruning``.
        For MC: mask step uses concrete multiply.
        """
        col_scores = self.token_scores[:n]

        # Pairwise diff → exp
        diff = raw_scores.unsqueeze(-2) - raw_scores.unsqueeze(-1)
        exp_diff = torch.exp(diff)

        if self.for_verification:
            # Heaviside: h(score_j) * exp(diff_ij)
            col_scores_j = col_scores.view(1, 1, 1, n).expand_as(exp_diff)
            exp_masked = heaviside_pruning(col_scores_j, exp_diff)
        else:
            col_mask = (col_scores >= 0).float()
            col_mask_j = col_mask.view(1, 1, 1, n).expand_as(exp_diff)
            exp_masked = exp_diff * col_mask_j

        # Sum over j → reciprocal
        sum_exp = exp_masked.sum(dim=-1)
        attn_w = torch.reciprocal(sum_exp)

        # Zero pruned output positions: h(score_k) * attn_w
        if self.for_verification:
            col_scores_k = col_scores.view(1, 1, n).expand_as(attn_w)
            attn_w = heaviside_pruning(col_scores_k, attn_w)
        else:
            col_mask_k = (col_scores >= 0).float().view(1, 1, n).expand_as(attn_w)
            attn_w = attn_w * col_mask_k

        return attn_w

    def forward(self, x: Tensor) -> Tensor:
        for layer_idx in range(self.n_layers):
            if layer_idx == self.mask_from_layer:
                x = self._apply_token_mask(x)

            n = x.shape[0]
            h = self.heads_list[layer_idx]
            d = self.dim_head_list[layer_idx]
            scale = self.scale_list[layer_idx]
            attn = self.attns[layer_idx]

            residual = x
            xn = self.attn_norms[layer_idx](x)
            q = attn.to_q(xn).reshape(n, h, d).permute(1, 0, 2)
            k = attn.to_k(xn).reshape(n, h, d).permute(1, 0, 2)
            v = attn.to_v(xn).reshape(n, h, d).permute(1, 0, 2)
            raw_scores = (q @ k.transpose(-2, -1)) * scale

            if layer_idx >= self.mask_from_layer:
                attn_w = self._masked_attention_weights(raw_scores, n)
            else:
                attn_w = raw_scores.softmax(dim=-1)

            out = (attn_w @ v).permute(1, 0, 2).reshape(n, h * d)
            out = attn.to_out(out)
            x = residual + out
            x = self.ff_blocks[layer_idx](x)

        x = x.mean(dim=0) if self.pool == "mean" else x[0]
        return self.mlp_head(x)