"""Certify differential bounds: trained SST-2 BERT vs quantized copy.

Loads the DeepT checkpoint, evaluates on real SST-2 validation sentences,
and averages certified bounds across samples.

Usage:
    caffeinate -i python3 experiments/certify_sst2_bert.py --layers 1 --bits 4 8 16
    caffeinate -i python3 experiments/certify_sst2_bert.py --layers 2 --bits 4 8 16
    caffeinate -i python3 experiments/certify_sst2_bert.py --layers 3 --bits 8 --no-tanh
    caffeinate -i python3 experiments/certify_sst2_bert.py --layers 3 --bits 8 --eps 0.001
    caffeinate -i python3 experiments/certify_sst2_bert.py --layers 1 --skip-diff
    caffeinate -i python3 experiments/certify_sst2_bert.py --layers 1 --max-len 10
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

import boundlab.expr as expr
import boundlab.prop as prop
import boundlab.zono as zono
from boundlab.diff.expr import DiffExpr3
from boundlab.diff.net import diff_net
from boundlab.diff.zono3 import interpret as diff_interpret
from boundlab.interp.onnx import onnx_export

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "model"


# ---------------------------------------------------------------------------
# Model (matches DeepT exactly)
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
    def __init__(self, d_model=128, d_ff=128, num_heads=4, num_layers=3,
                 num_classes=2, use_tanh=True):
        super().__init__()
        self.use_tanh = use_tanh
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads)
            for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        pooled = self.pooler(x.mean(dim=0))
        if self.use_tanh:
            pooled = torch.tanh(pooled)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_vocab(path):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab[line.rstrip("\n")] = idx
    return vocab


def load_checkpoint(num_layers=None, use_tanh=True):
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)

    state = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=False)
    n_layers = num_layers if num_layers is not None else config["num_hidden_layers"]

    model = DeepTBert(
        d_model=config["hidden_size"],
        d_ff=config["intermediate_size"],
        num_heads=config["num_attention_heads"],
        num_layers=n_layers,
        use_tanh=use_tanh,
    )

    with torch.no_grad():
        for i in range(n_layers):
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


# ---------------------------------------------------------------------------
# Load real SST-2 samples
# ---------------------------------------------------------------------------

def load_sst2_samples(model, state, vocab, n=10, max_len=32):
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


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_model(model, bits):
    model2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in model2.parameters():
            abs_max = p.abs().max()
            if abs_max == 0: continue
            scale = (2 ** (bits - 1) - 1) / abs_max
            p.copy_(torch.round(p * scale) / scale)
    return model2.eval()


# ---------------------------------------------------------------------------
# Verification methods
# ---------------------------------------------------------------------------

def certify_interval(model1, model2, center, eps):
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    shape = list(center.shape)
    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))
    x1 = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    x2 = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    ub1, lb1 = zono.interpret(gm1)(x1).ublb()
    ub2, lb2 = zono.interpret(gm2)(x2).ublb()
    return ub1 - lb2, lb1 - ub2


def certify_zonotope(model1, model2, center, eps):
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    shape = list(center.shape)
    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    d = zono.interpret(gm1)(x) - zono.interpret(gm2)(x)
    return d.ublb()


def certify_differential(model1, model2, center, eps):
    prop._UB_CACHE.clear(); prop._LB_CACHE.clear()
    shape = list(center.shape)
    gm1 = onnx_export(model1, (shape,))
    gm2 = onnx_export(model2, (shape,))
    merged = diff_net(gm1, gm2)
    op = diff_interpret(merged)
    x = expr.ConstVal(center) + eps * expr.LpEpsilon(shape)
    out = op(x)
    if isinstance(out, DiffExpr3):
        return out.diff.ublb()
    else:
        d = out.x - out.y
        return d.ublb()


def monte_carlo(model1, model2, center, eps, n_samples=5000):
    noise = (torch.rand(n_samples, *center.shape) * 2 - 1) * eps
    samples = center.unsqueeze(0) + noise
    with torch.no_grad():
        diffs = torch.stack([model1(s) - model2(s) for s in samples])
    return diffs


# ---------------------------------------------------------------------------
# Run one method safely
# ---------------------------------------------------------------------------

def run_method(fn, model1, model2, center, eps, mc_diffs):
    t0 = time.perf_counter()
    try:
        d_ub, d_lb = fn(model1, model2, center, eps)
        elapsed = time.perf_counter() - t0
        bound = max(d_ub.abs().max().item(), d_lb.abs().max().item())
        sound = (mc_diffs <= d_ub.unsqueeze(0) + 1e-4).all() and \
                (mc_diffs >= d_lb.unsqueeze(0) - 1e-4).all()
        width = (d_ub - d_lb).mean().item()
        return bound, elapsed, sound.item(), width
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return None, elapsed, False, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--bits", type=int, nargs="+", default=[8])
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=5000)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--skip-diff", action="store_true")
    parser.add_argument("--skip-zono", action="store_true")
    parser.add_argument("--no-tanh", action="store_true",
                        help="Remove tanh from pooler (avoids saturation at depth)")
    args = parser.parse_args()

    # Load checkpoint
    print("Loading checkpoint...")
    vocab = load_vocab(MODEL_DIR / "vocab.txt")
    model1, state, config = load_checkpoint(
        num_layers=args.layers, use_tanh=not args.no_tanh
    )
    n_params = sum(p.numel() for p in model1.parameters())

    # Load SST-2 samples
    print("Loading SST-2 validation samples...")
    samples = load_sst2_samples(model1, state, vocab, n=args.n_samples, max_len=args.max_len)
    print(f"Using {len(samples)} correctly classified samples")

    # Build method list
    methods = [("Int-Sub", certify_interval)]
    if not args.skip_zono:
        methods.append(("Zono-Sub", certify_zonotope))
    if not args.skip_diff:
        methods.append(("Differential", certify_differential))

    tanh_str = "no tanh" if args.no_tanh else "with tanh"
    print(f"\n{'='*80}")
    print(f"CERTIFY: SST-2 BERT ({args.layers} layers, {tanh_str}) vs Quantized")
    print(f"{'='*80}")
    print(f"  d_model=128, d_ff=128, heads=4, layers={args.layers}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Pooler: mean + {'tanh + ' if not args.no_tanh else ''}linear(2)")
    print(f"  input_eps={args.eps}, MC samples={args.mc_samples}")
    print(f"  Quantization bits: {args.bits}")
    print(f"  Max sentence length: {args.max_len} tokens")
    print(f"  Methods: {', '.join(n for n, _ in methods)}")

    # Run for each bit level
    for bits in args.bits:
        model2 = quantize_model(model1, bits)

        total_diff = sum((p1 - p2).abs().sum().item()
                         for p1, p2 in zip(model1.parameters(), model2.parameters()))
        print(f"\n{'='*80}")
        print(f"  {bits}-bit quantization (weight L1 diff: {total_diff:.4f})")
        print(f"{'='*80}")

        all_results = {name: [] for name, _ in methods}
        all_mc = []
        all_sound = {name: True for name, _ in methods}

        for i, (sent, center, tokens) in enumerate(samples):
            print(f"\n  [{i+1}/{len(samples)}] {sent[:60]}", flush=True)

            mc_diffs = monte_carlo(model1, model2, center, args.eps, args.mc_samples)
            mc_bound = mc_diffs.abs().max().item()
            all_mc.append(mc_bound)

            for name, fn in methods:
                print(f"    {name}...", end="", flush=True)
                bound, elapsed, sound, width = run_method(
                    fn, model1, model2, center, args.eps, mc_diffs)
                if bound is not None:
                    all_results[name].append((bound, elapsed, width))
                    print(f" bound={bound:.6f} time={elapsed:.1f}s sound={sound}")
                else:
                    print(f" FAILED ({elapsed:.1f}s)")
                if not sound:
                    all_sound[name] = False

            print(f"    MC={mc_bound:.6f}")

        # Summary
        avg_mc = sum(all_mc) / len(all_mc) if all_mc else 0

        print(f"\n  {'='*70}")
        print(f"  SUMMARY: {bits}-bit, {args.layers} layers, {len(samples)} samples, eps={args.eps}")
        print(f"  {'='*70}")
        print(f"  {'Method':<20} {'Avg Bound':>10} {'Avg Time':>9} {'Tight':>8} {'Sound':>6}")
        print(f"  {'-'*56}")
        print(f"  {'MC ground truth':<20} {avg_mc:>10.6f} {'---':>9} {'---':>8} {'---':>6}")

        for name, _ in methods:
            r = all_results[name]
            if r:
                avg_bound = sum(b for b, _, _ in r) / len(r)
                avg_time = sum(t for _, t, _ in r) / len(r)
                tight = f"{avg_mc/avg_bound:.1%}" if avg_bound > 0 else "---"
                snd = "Y" if all_sound[name] else "N"
                print(f"  {name:<20} {avg_bound:>10.6f} {avg_time:>8.1f}s {tight:>8} {snd:>6}")
            else:
                print(f"  {name:<20} {'FAIL':>10} {'---':>9} {'---':>8} {'N':>6}")

        int_results = all_results.get("Int-Sub", [])
        last_name = methods[-1][0]
        last_results = all_results.get(last_name, [])
        if int_results and last_results:
            avg_w_int = sum(w for _, _, w in int_results) / len(int_results)
            avg_w_last = sum(w for _, _, w in last_results) / len(last_results)
            if avg_w_int > 0:
                print(f"\n  {last_name} width reduction vs Int-Sub: {(1 - avg_w_last/avg_w_int)*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()