# MNIST ViT — Token Pruning Verification

Certifies that pruning tokens from a Vision Transformer doesn't change the
output by more than a verified bound, under L∞ input perturbation.

Three verification methods are compared:

| Method | What it computes | Guarantee |
|---|---|---|
| **MC** (Monte Carlo) | Sample perturbations, run concrete top-K | Empirical lower bound |
| **Zono-Sub** | Shared zonotope through both models, subtract | Sound upper bound |
| **Diff (Zono3)** | Differential zonotope via `diff_net` | Sound upper bound (tightest) |

Expected ordering: **MC ≤ Diff ≤ Zono-Sub**.


## Quick Start

```bash
# 1-layer ViT, default settings (eps=0.004, K=8, 3 samples)
python run_pruning.py

# 3-layer ViT
python run_pruning.py --depth 3 --checkpoint mnist_transformer_3.pt

# Custom settings
python run_pruning.py --eps 0.002 --K 12 --n-samples 10 --mc-samples 1000

# Method comparison with profiling (5 methods)
python pruning_zono.py --eps 0.002 --K 8 --n-samples 5

# Tests
pytest tests/ -v
```


## How It Works

### Verification Pipeline

```
  image (1,28,28) + ε                     token scores
        │                                      │
        ▼                                      ▼
  ┌─────────────┐    ┌───────────────┐   ┌───────────┐
  │ patch embed  │───►│ build_input_  │   │ classify_ │
  │ (ONNX)      │    │ zonotope      │   │ topk      │
  └─────────────┘    │ (PadOp+CLS+   │   └─────┬─────┘
                     │  pos_emb)     │         │
                     └───────┬───────┘   keep/prune/uncertain
                             │                 │
               zonotope (N+1, D)         ┌─────▼──────┐
                             │           │ enumerate_  │
                     ┌───────┴───────┐   │ cases      │
                     │               │   └─────┬──────┘
                     ▼               ▼         │
               PrunedViT       PrunedViT       │
               (full)          (pruned)    per case
                     │               │         │
                     ▼               ▼         │
                   diff_net ◄──────────────────┘
                     │
                 diff_interpret
                     │
                 DiffExpr3.diff.ublb()
                     │
                 union bounds over all cases
```

### Step by Step

1. **Build symbolic input** — `build_input_zonotope` runs the image through
   the patch embedding, prepends the CLS token via `PadOp` (not `Cat` — see
   "Why the Split Pipeline" below), and adds positional embedding.  Result:
   an `AffineSum` expression of shape `(N+1, D)`.

2. **Score tokens** — `ScoringModel` extracts CLS→patch attention importance.
   The zonotope interpreter propagates uncertainty through the scoring model,
   giving interval-bounded scores `[lb, ub]` per patch.

3. **Classify tokens** — `classify_topk` partitions patches into
   *definite-keep* (guaranteed top-K), *definite-prune* (guaranteed not
   top-K), and *uncertain* (could go either way under perturbation).

4. **Enumerate cases** — `enumerate_pruning_cases` generates all
   C(|uncertain|, K_remaining) possible kept-sets.

5. **Verify per case** — For each kept-set, export `PrunedViT` (full) and
   `PrunedViT` (pruned), merge via `diff_net`, run through `diff_interpret`,
   extract `DiffExpr3.diff.ublb()`.

6. **Union bounds** — Take element-wise max of ub, min of lb across all cases.

### Masked Softmax via Heaviside

Token pruning modifies attention: pruned tokens should get zero attention
weight.  Instead of a monolithic custom op, this is decomposed into
primitives the verifier already handles:

```
exp(s_j - s_k)                    ← exp linearizer
h(score_j) * exp(s_j - s_k)      ← heaviside_pruning (h = Heaviside step)
Σ_j [above]                       ← sum (affine)
1 / [above]                       ← reciprocal linearizer
h(score_k) * [above]              ← heaviside_pruning (zero pruned outputs)
```

`heaviside_pruning` is a registered ONNX op (`boundlab::HeavisidePruning`).
With ±large score constants from the case split, the Heaviside linearizer
produces exact 0/1 with zero approximation error — identical to a concrete
mask multiply.

### Why the Split Pipeline

`torch.cat(cls_token, patch_embeddings)` creates a `Cat` expression node.
`Cat` breaks `symmetric_decompose` in the bilinear matmul handler, loosening
bounds.  The workaround: export the patch embedding separately, then prepend
CLS via `PadOp + Add`.  This produces an `AffineSum` with symmetric
`LpEpsilon` children.  That's why `build_input_zonotope` exists instead of
exporting the whole ViT as one ONNX graph.


## File Map

### Core modules

| File | What it does |
|---|---|
| `mnist_vit.py` | ViT model definition. DeepT-checkpoint-compatible, no batch dim, no einops. `build_mnist_vit(checkpoint)` is the main entry point. |
| `pipeline.py` | Model wrappers for verification: `ScoringModel` and `PrunedViT`. |
| `token_pruning.py` | Pruning logic: score/mask construction, token classification, case enumeration, zonotope construction, ONNX export helpers, end-to-end certification. |
| `heaviside_handler.py` | Registers `HeavisidePruning` in the standard zonotope interpreter. The diff handler is already in core (`boundlab.diff.zono3`). |
| `masked_softmax.py` | Legacy `MaskedSoftmax` custom ONNX op and handlers. Being replaced by the Heaviside decomposition. |

### Scripts

| File | What it does |
|---|---|
| `run_pruning.py` | Benchmark: MC vs Zono-Sub vs Diff. Supports `--depth 1\|3`. |
| `pruning_zono.py` | Extended comparison of 5 methods with profiling and bound-width breakdown. |
| `export_to_onnx.py` | Saves patch embedding, scoring model, and pruned ViT as `.onnx` files. |

### Tests

| File | What it does |
|---|---|
| `tests/test_soundness.py` | Zonotope invariants, bounds enclosure, PGD cross-check. |

### Other

| File | What it does |
|---|---|
| `pruning.py` | Backwards-compatibility shim mapping old names to new modules. |
| `old_ver/` | Historical scripts. No active code imports from here. |
| `mnist_transformer.pt` | 1-layer DeepT MNIST checkpoint. |
| `mnist_transformer_3.pt` | 3-layer MNIST checkpoint. |


## Component Catalog

### `pipeline.py`

**`ScoringModel(vit, score_layer=0)`**
Extracts per-patch importance from CLS attention weights.
Input: `(N+1, D)` token embeddings → Output: `(N,)` importance scores.
`score_layer` controls which transformer layer's attention to use;
layers before it are run unmasked first.

**`PrunedViT(vit, token_scores, mask_from_layer=0, for_verification=False)`**
Runs the ViT transformer + classification head with token pruning.
Input: `(N+1, D)` token embeddings → Output: `(num_classes,)` logits.
`token_scores` is `(N+1,)`: positive → kept, negative → pruned.
`for_verification=True` uses `heaviside_pruning` ONNX ops (for export).
`for_verification=False` uses concrete masking (for Monte Carlo).

### `token_pruning.py`

**`build_token_scores(num_tokens, kept_patches, magnitude=100.0)`**
Returns `(N+1,)` scores: `+magnitude` for kept, `-magnitude` for pruned.
CLS (index 0) always kept.

**`build_all_kept_scores(num_tokens)`**
All tokens kept (no pruning). Convenience wrapper.

**`classify_topk(ub_scores, lb_scores, K)`**
Given interval-bounded scores, partitions N patches into `definite_keep`,
`definite_prune`, `uncertain`. Returns three sets of patch indices.

**`enumerate_pruning_cases(definite_keep, uncertain, K)`**
Generates all C(|uncertain|, K_remaining) valid kept-sets.

**`build_input_zonotope(vit, img, eps, op_patch)`**
Builds the symbolic zonotope for the full token sequence from an image.
Uses PadOp to prepend CLS (avoids Cat node).

**`export_patch_embedding(vit, img_shape)`**
ONNX-exports `vit.to_patch_embedding`. Returns a zonotope interpreter.

**`export_scoring(vit, num_tokens, dim, score_layer=0)`**
ONNX-exports `ScoringModel`. Returns `(interpreter, concrete_model)`.

**`export_pruned_vit(vit, kept_patches, num_tokens, dim, mask_from_layer=0)`**
ONNX-exports a `PrunedViT` with `for_verification=True`.
Returns an ONNX IR model for `zono.interpret` or `diff_net`.

**`certify_pruning_diff(vit, img, eps, K, op_patch, op_score, ...)`**
End-to-end certification: build zonotope → score → classify → case-split
→ diff verify → union.  Returns `CertifyResult(ub, lb, n_cases, ...)`.

### `heaviside_handler.py`

**`heaviside_zono_handler(scores, data)`**
Standard zonotope handler for `h(scores) * data`.
Concrete scores → exact 0/1 mask. Symbolic scores → `_linearize_hsx`.
Registered as `zono.interpret["HeavisidePruning"]` at import time.

### `masked_softmax.py` *(transitional)*

**`masked_softmax_op(scores, col_mask)`**
Custom ONNX op. Returns zeros at runtime; creates `boundlab::MaskedSoftmax`
node for verification.

**`masked_softmax_zono_handler(x, col_mask)`**
Zonotope handler. DeepT decomposition with concrete mask multiplication.

**`diff_masked_softmax_handler(x, col_mask)`**
Differential handler. Same decomposition on `DiffExpr3`.

### `mnist_vit.py`

**`ViT(...)`** — Vision Transformer. No batch dim: `(C,H,W) → (num_classes,)`.
**`build_mnist_vit(checkpoint)`** — Builds the 1-layer MNIST ViT from a DeepT checkpoint.


## ONNX

The ONNX graph produced by `export_pruned_vit` contains standard ops plus
`boundlab::HeavisidePruning` (a custom-domain op).  BoundLab's interpreters
handle it; external ONNX runtimes will not.

To save ONNX files to disk:

```bash
python export_to_onnx.py
python export_to_onnx.py --depth 3 --checkpoint mnist_transformer_3.pt
python export_to_onnx.py --K 12 --out-dir ./onnx_models
```

This exports four files: `patch_embedding.onnx`, `scoring_model.onnx`,
`pruned_vit_full.onnx`, and `pruned_vit_k8.onnx`.