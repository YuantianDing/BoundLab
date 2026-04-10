"""Compare BoundLab vs DeepT on a 3-layer SST transformer (LayerNorm removed).

This script loads weights from DeepT's ``sst_bert_small_3/ckpt-3`` checkpoint,
constructs a 3-layer transformer with attention + FFN blocks (no LayerNorm), and
compares output bounds under an L∞ perturbation on one token embedding.

Run:
    pixi run python compare/deept/compare.py
"""

from __future__ import annotations

import math
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

import boundlab.expr as expr
import boundlab.zono as zono
from boundlab.interp.onnx import onnx_export
from Zonotope import Zonotope as DeepTZonotope  # type: ignore
from Zonotope import make_zonotope_new_weights_same_args  # type: ignore


HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "DeepT" / "Robustness-Verification-for-Transformers" / "sst_bert_small_3" / "ckpt-3"
CKPT_PATH = CKPT_DIR / "pytorch_model.bin"
VOCAB_PATH = CKPT_DIR / "vocab.txt"


def _deept_args(input_dim: int, device: str = "cpu") -> Namespace:
    return Namespace(
        perturbed_words=1,
        attack_type="l_inf",
        all_words=False,
        device=device,
        cpu=(device == "cpu"),
        num_input_error_terms=input_dim,
        zonotope_slow=False,
        error_reduction_method="None",
        p=float("inf"),
        add_softmax_sum_constraint=False,
        use_dot_product_variant3=False,
        use_other_dot_product_ordering=False,
        num_perturbed_words=1,
        concretize_special_norm_error_together=False,
        batch_softmax_computation=False,
        keep_intermediate_zonotopes=False,
    )


def _linear_from_state(state: dict[str, torch.Tensor], prefix: str) -> nn.Linear:
    w = state[f"{prefix}.weight"]
    b = state[f"{prefix}.bias"]
    layer = nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        layer.weight.copy_(w)
        layer.bias.copy_(b)
    return layer


class SstSmall3NoLayerNorm(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor], num_layers: int = 3, num_heads: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden = int(state["bert.pooler.dense.weight"].shape[1])
        assert self.hidden % num_heads == 0
        self.head_dim = self.hidden // num_heads

        self.query = nn.ModuleList([
            _linear_from_state(state, f"bert.encoder.layer.{i}.attention.self.query")
            for i in range(num_layers)
        ])
        self.key = nn.ModuleList([
            _linear_from_state(state, f"bert.encoder.layer.{i}.attention.self.key")
            for i in range(num_layers)
        ])
        self.value = nn.ModuleList([
            _linear_from_state(state, f"bert.encoder.layer.{i}.attention.self.value")
            for i in range(num_layers)
        ])
        self.attn_out = nn.ModuleList([
            _linear_from_state(state, f"bert.encoder.layer.{i}.attention.output.dense")
            for i in range(num_layers)
        ])
        self.ff_in = nn.ModuleList([
            _linear_from_state(state, f"bert.encoder.layer.{i}.intermediate.dense")
            for i in range(num_layers)
        ])
        self.ff_out = nn.ModuleList([
            _linear_from_state(state, f"bert.encoder.layer.{i}.output.dense")
            for i in range(num_layers)
        ])

        self.pooler = _linear_from_state(state, "bert.pooler.dense")
        self.classifier = _linear_from_state(state, "classifier")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, hidden]
        for i in range(self.num_layers):
            q = self.query[i](x)
            k = self.key[i](x)
            v = self.value[i](x)

            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden)
            probs = torch.softmax(scores, dim=-1)
            context = torch.matmul(probs, v)

            x_attn = x + self.attn_out[i](context)
            x = x_attn + self.ff_out[i](torch.relu(self.ff_in[i](x_attn)))

        pooled = torch.tanh(self.pooler(x.mean(dim=0)))
        return self.classifier(pooled)


def _load_vocab(path: Path) -> dict[str, int]:
    vocab = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab[line.rstrip("\n")] = idx
    return vocab


def build_input_from_embeddings(
    state: dict[str, torch.Tensor],
    vocab: dict[str, int],
    tokens: list[str],
) -> tuple[torch.Tensor, int]:
    ids = [vocab.get(tok, vocab.get("[UNK]", 100)) for tok in tokens]
    word = state["bert.embeddings.word_embeddings.weight"][ids]
    pos = state["bert.embeddings.position_embeddings.weight"][: len(ids)]
    tok = state["bert.embeddings.token_type_embeddings.weight"][0].unsqueeze(0).expand(len(ids), -1)
    center = word + pos + tok

    perturbed_idx = tokens.index("good") if "good" in tokens else 1
    return center, perturbed_idx


def boundlab_bounds(model: nn.Module, center: torch.Tensor, perturbed_idx: int, eps: float):
    exported = onnx_export(model, (center,))
    op_types = sorted({node.op_type for node in exported.graph})
    unsupported = [op for op in op_types if op not in zono.interpret]
    if unsupported:
        raise RuntimeError(f"Unsupported ONNX ops for zono.interpret: {unsupported}")

    op = zono.interpret(exported)
    noise = expr.LpEpsilon(list(center.shape), p="inf")
    mask = torch.zeros_like(center)
    mask[perturbed_idx] = 1.0
    x = center + eps * (noise * mask)
    y = op(x)
    ub, lb = y.ublb()
    return ub, lb, op_types


def deept_bounds(model: SstSmall3NoLayerNorm, center: torch.Tensor, perturbed_idx: int, eps: float):
    args = _deept_args(input_dim=center.shape[1])
    z = DeepTZonotope(args, p=float("inf"), eps=eps, perturbed_word_index=perturbed_idx, value=center)

    for i in range(model.num_layers):
        q = z.dense(model.query[i]).add_attention_heads_dim(model.num_heads)
        k = z.dense(model.key[i]).add_attention_heads_dim(model.num_heads)
        scores = q.dot_product(k).multiply(1.0 / math.sqrt(model.hidden))
        probs = scores.softmax(no_constraints=True)

        v = z.dense(model.value[i]).add_attention_heads_dim(model.num_heads)
        context = probs.dot_product(v.t()).remove_attention_heads_dim()

        x_attn = context.dense(model.attn_out[i])
        z_res = z.expand_error_terms_to_match_zonotope(x_attn)
        x_attn = x_attn.add(z_res)
        h = x_attn.dense(model.ff_in[i]).relu()
        z_next = h.dense(model.ff_out[i])
        x_res = x_attn.expand_error_terms_to_match_zonotope(z_next)
        z = z_next.add(x_res)

    pooled_w = z.zonotope_w.mean(dim=1, keepdim=True)
    pooled = make_zonotope_new_weights_same_args(new_weights=pooled_w, source_zonotope=z, clone=False)
    pooled = pooled.dense(model.pooler).tanh()
    logits = pooled.dense(model.classifier)

    lb, ub = logits.concretize()
    return ub.squeeze(), lb.squeeze()


def main() -> None:
    state = torch.load(CKPT_PATH, map_location="cpu")
    vocab = _load_vocab(VOCAB_PATH)

    model = SstSmall3NoLayerNorm(state, num_layers=3, num_heads=1).eval()
    tokens = ["[CLS]", "a", "very", "good", "movie", "[SEP]"]
    center, perturbed_idx = build_input_from_embeddings(state, vocab, tokens)
    eps = 0.01

    t0 = time.perf_counter()
    bl_ub, bl_lb, op_types = boundlab_bounds(model, center, perturbed_idx, eps)
    bl_time = time.perf_counter() - t0
    max_bl_width = torch.max(bl_ub - bl_lb).item()

    t0 = time.perf_counter()
    dt_ub, dt_lb = deept_bounds(model, center, perturbed_idx, eps)
    dt_time = time.perf_counter() - t0
    max_dt_width = torch.max(dt_ub - dt_lb).item()

    print("Model: sst_bert_small_3 (3-layer, LayerNorm removed)")
    print(f"Sentence tokens: {tokens}")
    print(f"Perturbed token index: {perturbed_idx} ({tokens[perturbed_idx]})")
    print(f"Epsilon (L_inf): {eps}")
    print(f"ONNX ops ({len(op_types)}): {', '.join(op_types)}")
    print()
    print(f"BoundLab width: {max_bl_width:.6f}")
    print(f"DeepT width:    {max_dt_width:.6f}")
    print()
    print(f"Time BoundLab: {bl_time:.3f}s")
    print(f"Time DeepT:    {dt_time:.3f}s")
    print(f"DeepT/BoundLab: {(dt_time / bl_time if bl_time > 0 else float('inf')):.2f}x")


if __name__ == "__main__":
    main()
