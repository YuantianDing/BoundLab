"""BERT SST-2 benchmark: differential vs zonotope subtraction.

3-layer BERT, 4/8/16-bit quantization, eps=0.005/0.01/0.03/0.05, 100 samples.

Usage:
    caffeinate python experiments/run_bert.py
    caffeinate python experiments/run_bert.py --n-samples 10   # quick test
"""
from __future__ import annotations
import sys, time, warnings, os, copy, json, math
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import torch.nn as nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAYERS = 3
BITS_LIST = [8, 16]
EPS_LIST = [0.0025, 0.005, 0.01, 0.02]
N_SAMPLES = 100

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "experiments" else HERE
MODEL_DIR = ROOT / os.environ.get("BOUNDLAB_MODEL_DIR", "model")


class _Quiet:
    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self._devnull, self._devnull
        return self
    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._old_out, self._old_err
        self._devnull.close()


# ---------------------------------------------------------------------------
# Model (DeepT BERT)
# ---------------------------------------------------------------------------

class LayerNormNoVar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        S, _ = x.shape
        h = self.num_heads
        Q = self.query(x).reshape(S, h, -1).permute(1, 0, 2)
        K = self.key(x).reshape(S, h, -1).permute(1, 0, 2)
        V = self.value(x).reshape(S, h, -1).permute(1, 0, 2)
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        attn = scores.softmax(dim=-1)
        context = (attn @ V).permute(1, 0, 2).reshape(S, -1)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.attn_norm = LayerNormNoVar(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ff_norm = LayerNormNoVar(d_model)
    def forward(self, x):
        x = self.attn_norm(x + self.attn(x))
        x = self.ff_norm(x + self.ff2(self.relu(self.ff1(x))))
        return x


class DeepTBert(nn.Module):
    def __init__(self, d_model=128, d_ff=128, num_heads=4, num_layers=3, num_classes=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads) for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        pooled = torch.tanh(self.pooler(x.mean(dim=0)))
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_vocab(path):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab[line.rstrip("\n")] = idx
    return vocab


def load_checkpoint(num_layers):
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)
    state = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=False)
    model = DeepTBert(
        d_model=config["hidden_size"], d_ff=config["intermediate_size"],
        num_heads=config["num_attention_heads"], num_layers=num_layers,
    )
    with torch.no_grad():
        for i in range(num_layers):
            b = model.blocks[i]
            p = f"bert.encoder.layer.{i}"
            b.attn.query.weight.copy_(state[f"{p}.attention.self.query.weight"])
            b.attn.query.bias.copy_(state[f"{p}.attention.self.query.bias"])
            b.attn.key.weight.copy_(state[f"{p}.attention.self.key.weight"])
            b.attn.key.bias.copy_(state[f"{p}.attention.self.key.bias"])
            b.attn.value.weight.copy_(state[f"{p}.attention.self.value.weight"])
            b.attn.value.bias.copy_(state[f"{p}.attention.self.value.bias"])
            b.attn.out_proj.weight.copy_(state[f"{p}.attention.output.dense.weight"])
            b.attn.out_proj.bias.copy_(state[f"{p}.attention.output.dense.bias"])
            b.attn_norm.weight.copy_(state[f"{p}.attention.output.LayerNorm.weight"])
            b.attn_norm.bias.copy_(state[f"{p}.attention.output.LayerNorm.bias"])
            b.ff1.weight.copy_(state[f"{p}.intermediate.dense.weight"])
            b.ff1.bias.copy_(state[f"{p}.intermediate.dense.bias"])
            b.ff2.weight.copy_(state[f"{p}.output.dense.weight"])
            b.ff2.bias.copy_(state[f"{p}.output.dense.bias"])
            b.ff_norm.weight.copy_(state[f"{p}.output.LayerNorm.weight"])
            b.ff_norm.bias.copy_(state[f"{p}.output.LayerNorm.bias"])
        model.pooler.weight.copy_(state["bert.pooler.dense.weight"])
        model.pooler.bias.copy_(state["bert.pooler.dense.bias"])
        model.classifier.weight.copy_(state["classifier.weight"])
        model.classifier.bias.copy_(state["classifier.bias"])
    return model.eval(), state, config


def embed_sentence(state, vocab, sentence):
    tokens = ["[CLS]"] + sentence.lower().split() + ["[SEP]"]
    unk_id = vocab.get("[UNK]", 100)
    ids = [vocab.get(tok, unk_id) for tok in tokens]
    ids_t = torch.tensor(ids, dtype=torch.long)
    word = state["bert.embeddings.word_embeddings.weight"][ids_t]
    pos = state["bert.embeddings.position_embeddings.weight"][:len(ids)]
    tok = state["bert.embeddings.token_type_embeddings.weight"][0].unsqueeze(0).expand(len(ids), -1)
    emb = word + pos + tok
    ln_w = state["bert.embeddings.LayerNorm.weight"]
    ln_b = state["bert.embeddings.LayerNorm.bias"]
    emb = ln_w * (emb - emb.mean(-1, keepdim=True)) + ln_b
    return emb, tokens


def load_sst2_samples(model, state, vocab, n, max_len=32):
    from datasets import load_dataset
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    samples = []
    for row in sst:
        if len(samples) >= n:
            break
        sent = row["sentence"]
        tokens = ["[CLS]"] + sent.lower().split() + ["[SEP]"]
        if len(tokens) > max_len:
            continue
        center, tok_list = embed_sentence(state, vocab, sent)
        with torch.no_grad():
            logits = model(center)
            pred = logits.argmax().item()
        if pred == row["label"]:
            samples.append((sent, center, tok_list))
    return samples


def quantize_model(model, bits):
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            abs_max = p.abs().max()
            if abs_max == 0:
                continue
            scale = (2 ** (bits - 1) - 1) / abs_max
            p.copy_(torch.round(p * scale) / scale)
    return model2.eval()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_diff(model1, model2, center, eps):
    with _Quiet():
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        shape = list(center.shape)
        gm1 = onnx_export(model1, (shape,))
        gm2 = onnx_export(model2, (shape,))
        merged = diff_net(gm1, gm2)
        op = diff_interpret(merged)
        x = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
        out = op(x)
        if isinstance(out, DiffExpr3):
            ub, lb = out.diff.ublb()
        else:
            ub, lb = (out.x - out.y).ublb()
    return max(ub.abs().max().item(), lb.abs().max().item())


def verify_zono(model1, model2, center, eps):
    with _Quiet():
        prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
        shape = list(center.shape)
        gm1 = onnx_export(model1, (shape,))
        gm2 = onnx_export(model2, (shape,))
        x = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
        d = zono.interpret(gm1)(x) - zono.interpret(gm2)(x)
        ub, lb = d.ublb()
    return max(ub.abs().max().item(), lb.abs().max().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--bits", type=int, nargs="+", default=BITS_LIST)
    ap.add_argument("--eps-list", type=float, nargs="+", default=EPS_LIST)
    ap.add_argument("--max-len", type=int, default=12)
    args = ap.parse_args()

    print("Loading checkpoint...", flush=True)
    vocab = load_vocab(MODEL_DIR / "vocab.txt")
    model1, state, config = load_checkpoint(args.layers)
    n_params = sum(p.numel() for p in model1.parameters())

    print(f"Loading {args.n_samples} SST-2 samples (max_len={args.max_len})...", flush=True)
    samples = load_sst2_samples(model1, state, vocab, n=args.n_samples, max_len=args.max_len)
    print(f"  Got {len(samples)} correctly classified samples")

    print(f"\nBERT {args.layers}-layer ({n_params:,} params)")
    print(f"Bits: {args.bits}")
    print(f"Eps:  {args.eps_list}")
    print(f"Samples: {len(samples)}\n")

    grand_t0 = time.perf_counter()
    summary = {}  # (bits, eps) -> (mean_diff, mean_zono)

    for bits in args.bits:
        model2 = quantize_model(model1, bits)
        wdiff = sum((p1 - p2).abs().sum().item()
                     for p1, p2 in zip(model1.parameters(), model2.parameters()))

        for eps in args.eps_list:
            print(f"bits={bits}  eps={eps}  (weight L1 diff: {wdiff:.4f})")
            print(f"{'#':>4}  {'sent':>40}  {'D_diff':>10}  {'D_zono':>10}  "
                  f"{'gain':>6}  {'time':>7}")
            print("-" * 90)

            all_diff, all_zono = [], []
            t_total = time.perf_counter()

            for i, (sent, center, tokens) in enumerate(samples):
                t0 = time.perf_counter()

                try:
                    D_diff = verify_diff(model1, model2, center, eps)
                except Exception:
                    D_diff = float("inf")

                try:
                    D_zono = verify_zono(model1, model2, center, eps)
                except Exception:
                    D_zono = float("inf")

                elapsed = time.perf_counter() - t0
                all_diff.append(D_diff)
                all_zono.append(D_zono)
                if D_diff < float("inf") and D_zono < float("inf") and D_diff > 0:
                    gain_str = f"{D_zono / D_diff:5.1f}x"
                else:
                    gain_str = "  N/A"

                short = sent[:38] + ".." if len(sent) > 40 else sent
                print(f"{i+1:4d}  {short:>40}  {D_diff:10.4f}  {D_zono:10.4f}  "
                      f"{gain_str:>6}  {elapsed:6.1f}s", flush=True)

            print("-" * 90)
            td = torch.tensor(all_diff)
            tz = torch.tensor(all_zono)
            print(f"{'':>46}  {'Diff':>10}  {'Zono':>10}  {'Gain':>6}")
            print(f"  Mean{'':>40}  {td.mean().item():10.4f}  {tz.mean().item():10.4f}  "
                  f"{tz.mean().item()/(td.mean().item()+1e-30):5.1f}x")
            print(f"  Median{'':>38}  {td.median().item():10.4f}  {tz.median().item():10.4f}  "
                  f"{tz.median().item()/(td.median().item()+1e-30):5.1f}x")
            print(f"  Min{'':>41}  {td.min().item():10.4f}  {tz.min().item():10.4f}")
            print(f"  Max{'':>41}  {td.max().item():10.4f}  {tz.max().item():10.4f}")
            print(f"  Total time: {time.perf_counter() - t_total:.0f}s\n")

            summary[(bits, eps)] = (td.mean().item(), tz.mean().item())

    # Cross-table summary
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    header = f"  {'bits':>4} |" + "".join(f" {e:>10}" for e in args.eps_list)

    print(f"\n  Mean D (Differential):")
    print(header)
    print(f"  -----+" + "-" * (11 * len(args.eps_list)))
    for bits in args.bits:
        row = f"  {bits:>4} |"
        for eps in args.eps_list:
            if (bits, eps) in summary:
                row += f" {summary[(bits, eps)][0]:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print(f"\n  Mean D (Zonotope Sub):")
    print(header)
    print(f"  -----+" + "-" * (11 * len(args.eps_list)))
    for bits in args.bits:
        row = f"  {bits:>4} |"
        for eps in args.eps_list:
            if (bits, eps) in summary:
                row += f" {summary[(bits, eps)][1]:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print(f"\n  Gain (Zono / Diff):")
    print(header)
    print(f"  -----+" + "-" * (11 * len(args.eps_list)))
    for bits in args.bits:
        row = f"  {bits:>4} |"
        for eps in args.eps_list:
            if (bits, eps) in summary:
                d, z = summary[(bits, eps)]
                row += f" {z/(d+1e-30):>9.1f}x"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print(f"\nGrand total: {time.perf_counter() - grand_t0:.0f}s "
          f"({(time.perf_counter() - grand_t0)/60:.1f}m)")


if __name__ == "__main__":
    main()