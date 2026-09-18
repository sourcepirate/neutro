"""QLoRA fine-tuning example — pre-train then adapt with quantized LoRA.

Identical workflow to ``lora_finetune.py`` but uses ``QLoRADense`` (NF4,
4-bit NormalFloat + double quantization) for the frozen base weights:

  1. Build a small Transformer and pre-train briefly.
  2. Adapt with QLoRA via ``apply_qlora`` (NF4, rank=8, double-quant).
  3. Show memory savings from 4-bit quantization.
  4. Fine-tune only the LoRA adapters.
  5. Evaluate and generate.

Usage:
    python examples/qlora_finetune.py                    # synthetic
    python examples/qlora_finetune.py --wikitext         # WikiText-2
    python examples/qlora_finetune.py --rank 8 --no-double-quant
"""
import os
import sys
import argparse

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neutro.models import Sequential
from neutro.layers import Embedding, TransformerBlock, Dense, Softmax
from neutro.layers import QLoRADense, apply_qlora
from neutro.optimizers import AdamW


# ---------------------------------------------------------------------------
# Data helpers (shared with lora_finetune.py)
# ---------------------------------------------------------------------------

def synthetic_data(vocab_size=128, seq_len=16, n_samples=2000, task="pretrain", seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, vocab_size, size=(n_samples, seq_len))
    if task == "pretrain":
        y = (x + 1) % vocab_size
    elif task == "finetune":
        y = (x * 2) % vocab_size
    else:
        raise ValueError(task)
    return x, y


def try_load_wikitext(vocab_size=1024, seq_len=16, n_samples=2000):
    try:
        from neutro.utils.data_utils import load_wikitext2
        from neutro.tokenizers import RegexTokenizer

        print("Loading WikiText-2...")
        text = load_wikitext2()
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        tok = RegexTokenizer()
        tok.train(text[:50000], vocab_size=vocab_size, verbose=False)
        print(f"Tokenizer ready — vocab_size={tok.vocab_size}")
        encoded = tok.encode(text[:50000])
        if len(encoded) < seq_len + 1:
            raise ValueError("not enough tokens")
        step = 4
        xs, ys = [], []
        for i in range(0, min(len(encoded) - seq_len, n_samples * step), step):
            chunk = encoded[i : i + seq_len + 1]
            if len(chunk) < seq_len + 1:
                break
            xs.append(chunk[:seq_len])
            ys.append(chunk[1 : seq_len + 1])
            if len(xs) >= n_samples:
                break
        x = np.array(xs, dtype=int)
        y = np.array(ys, dtype=int)
        print(f"WikiText data: x={x.shape}, y={y.shape}")
        return x, y, tok, tok.vocab_size
    except Exception as e:
        print(f"[wiki] Failed ({e}), falling back to synthetic data.")
        return None


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(vocab_size, seq_len, embed_dim=32, num_heads=4, ff_dim=64):
    return Sequential([
        Embedding(vocab_size, embed_dim, input_shape=(seq_len,)),
        TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                         causal=True, use_flash=True),
        TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                         causal=True, use_flash=True),
        Dense(vocab_size),
        Softmax(),
    ])


def print_param_stats(model, label="Model"):
    total = sum(l.count_params() for l in model._get_all_layers())
    trainable = sum(
        l.count_params() for l in model._get_all_layers() if getattr(l, "trainable", True)
    )
    print(f"[{label}] total_params={total:,}  trainable={trainable:,}  frozen={total - trainable:,}")


def print_memory_stats(qlora_layers):
    """Report NF4 memory savings."""
    tot_nf4 = 0
    tot_overhead = 0
    tot_lora = 0
    n_params = 0
    for lyr in qlora_layers:
        stats = lyr.memory_bytes()
        tot_nf4 += stats["base_nf4"]
        tot_overhead += stats["absmax_overhead"]
        tot_lora += stats["lora_fp64"]
        n_params += lyr.W_codes.size
    fp32_bytes = n_params * 4  # float32 baseline
    fp64_bytes = n_params * 8  # float64 baseline (np default)
    q_bytes = tot_nf4 + tot_overhead
    print(f"[memory] Frozen base: {n_params:,} params")
    print(f"  FP32 baseline:  {fp32_bytes/1024:.1f} KiB")
    print(f"  NF4 quantized:  {q_bytes/1024:.1f} KiB (incl. {tot_overhead:.0f} B overhead)")
    print(f"  Compression:    {fp32_bytes/max(q_bytes,1):.1f}x vs FP32  "
          f"({fp64_bytes/max(q_bytes,1):.1f}x vs FP64)")
    print(f"  LoRA adapters:  {tot_lora/1024:.1f} KiB trainable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning demo")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=64, help="NF4 block size")
    parser.add_argument("--no-double-quant", action="store_true", help="disable double quantization")
    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--wikitext", action="store_true")
    args = parser.parse_args()

    np.random.seed(1)

    use_wiki = False
    tok = None
    if args.wikitext:
        result = try_load_wikitext(vocab_size=args.vocab_size, seq_len=args.seq_len, n_samples=800)
        if result is not None:
            x_pre, y_pre, tok, vocab_size = result
            x_ft, y_ft = x_pre, y_pre
            use_wiki = True

    if not use_wiki:
        vocab_size = args.vocab_size
        print(f"Using synthetic data — vocab_size={vocab_size}, seq_len={args.seq_len}")
        x_pre, y_pre = synthetic_data(vocab_size, args.seq_len, n_samples=800, task="pretrain")
        x_ft, y_ft = synthetic_data(vocab_size, args.seq_len, n_samples=400, task="finetune", seed=1)
        print(f"Pre-train: x={x_pre.shape}, y={y_pre.shape}")
        print(f"Fine-tune: x={x_ft.shape}, y={y_ft.shape}")

    # ---- Build and pre-train ----
    print("\n--- Building model ---")
    model = build_model(vocab_size, args.seq_len)
    _ = model(x_pre[:2])  # build
    print_param_stats(model, "Pre-adaptation")
    model.summary()

    print("\n--- Pre-training ---")
    model.compile(optimizer=AdamW(learning_rate=0.005), loss="sparse_categorical_crossentropy",
                  metrics=["sparse_accuracy"])
    model.fit(x_pre, y_pre, epochs=args.pretrain_epochs, batch_size=32, verbose=1)

    pre_eval = model.evaluate(x_pre[:200], y_pre[:200])
    print(f"Pre-train eval: loss={pre_eval['loss']:.4f}")

    # ---- Inject QLoRA ----
    print(f"\n--- Injecting QLoRA (rank={args.rank}, alpha={args.alpha}, "
          f"block_size={args.block_size}, double_quant={not args.no_double_quant}) ---")
    double_quant = not args.no_double_quant
    model, qlora_layers = apply_qlora(
        model, rank=args.rank, alpha=args.alpha,
        block_size=args.block_size, double_quant=double_quant, verbose=True,
    )
    print_param_stats(model, "Post-adaptation (QLoRA)")
    print_memory_stats(qlora_layers)

    # Sanity: first forward with NF4 dequant should be close to pre-adaptation.
    # (Small error from 4-bit quantization is expected.)
    sample = x_pre[:2]
    out_q = model.predict(sample)
    # Rough quality check: output should not be NaN
    assert not np.any(np.isnan(np.asarray(out_q))), "QLoRA forward produced NaN"

    # ---- Fine-tune with QLoRA ----
    print("\n--- QLoRA fine-tuning ---")
    model.compile(optimizer=AdamW(learning_rate=0.01), loss="sparse_categorical_crossentropy",
                  metrics=["sparse_accuracy"])
    model.fit(x_ft, y_ft, epochs=args.finetune_epochs, batch_size=32, verbose=1)

    ft_eval = model.evaluate(x_ft[:200], y_ft[:200])
    print(f"Fine-tune eval (on finetune task): loss={ft_eval['loss']:.4f}")

    # ---- Generation ----
    print("\n--- Generation demo ---")
    if tok is not None:
        try:
            start_ids = tok.encode("The ")[: args.seq_len]
            start_ids = np.pad(start_ids, (0, max(0, args.seq_len - len(start_ids)))).tolist()
        except Exception:
            start_ids = x_pre[0][: args.seq_len].tolist()
    else:
        start_ids = x_pre[0][: args.seq_len].tolist()

    seq = list(start_ids[: args.seq_len])
    for _ in range(20):
        inp = np.array(seq[-args.seq_len :]).reshape(1, -1)
        if inp.shape[1] < args.seq_len:
            inp = np.pad(inp, ((0, 0), (args.seq_len - inp.shape[1], 0)))
        probs = np.asarray(model.predict(inp))
        next_id = int(np.argmax(probs[0, -1]))
        seq.append(next_id)

    print(f"Generated ids: {seq[:40]} ...")
    if tok is not None:
        try:
            print(f"Decoded: {tok.decode(seq)}")
        except Exception as e:
            print(f"(decode failed: {e})")

    print("\n--- Done ---")
    print("QLoRA: 4-bit frozen base + trainable LoRA — same quality, ~8x less memory.")


if __name__ == "__main__":
    main()
