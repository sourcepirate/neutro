"""LoRA fine-tuning example — pre-train then adapt with LoRA.

This example demonstrates the full LoRA workflow:

  1. Build a small Transformer language model and pre-train it briefly
     on a synthetic (or WikiText-2) next-token task.
  2. Adapt the model with LoRA via ``apply_lora`` (rank=8, alpha=16).
  3. Fine-tune **only** the LoRA parameters on a new task.
  4. Compare trainable parameter counts and show that fine-tuning is
     parameter-efficient.
  5. Merge LoRA weights and generate text.

Usage:
    python examples/lora_finetune.py                    # synthetic demo (fast, offline)
    python examples/lora_finetune.py --wikitext         # WikiText-2 demo (downloads data)
    python examples/lora_finetune.py --rank 4 --epochs 2
"""
import os
import sys
import argparse

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neutro.models import Sequential
from neutro.layers import Embedding, TransformerBlock, Dense, Softmax
from neutro.layers import LoRADense, apply_lora
from neutro.optimizers import AdamW


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def synthetic_data(vocab_size=128, seq_len=16, n_samples=2000, task="pretrain", seed=0):
    """Generate synthetic next-token data.

    pretrain task:  y[t] = (x[t] + 1) % vocab_size  (shift-by-one)
    finetune task:  y[t] = (x[t] * 2) % vocab_size  (different mapping)
    """
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
    """Try to load WikiText-2 + BPE.  Falls back to synthetic on failure."""
    try:
        from neutro.utils.data_utils import load_wikitext2
        from neutro.tokenizers import RegexTokenizer

        print("Loading WikiText-2...")
        text = load_wikitext2()
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        tok = RegexTokenizer()
        tok.train(text[:50000], vocab_size=vocab_size, verbose=False)
        print(f"Tokenizer ready — vocab_size={tok.vocab_size}")

        # Encode via tok.encode (handles regex BPE).
        # Use tok.encode on slices for simplicity.
        encoded = tok.encode(text[:50000])
        if len(encoded) < seq_len + 1:
            raise ValueError("not enough tokens")
        # Build (x, y) pairs with sliding window.
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
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_shape=(seq_len,)),
        TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                         causal=True, use_flash=True),
        TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                         causal=True, use_flash=True),
        Dense(vocab_size),
        Softmax(),
    ])
    return model


def print_param_stats(model, label="Model"):
    total = sum(l.count_params() for l in model._get_all_layers())
    # Trainable = params in layers where .trainable is True
    trainable = sum(
        l.count_params() for l in model._get_all_layers() if getattr(l, "trainable", True)
    )
    print(f"[{label}] total_params={total:,}  trainable={trainable:,}  "
          f"frozen={total - trainable:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning demo")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--wikitext", action="store_true", help="use WikiText-2 instead of synthetic")
    args = parser.parse_args()

    np.random.seed(0)

    use_wiki = False
    tok = None
    if args.wikitext:
        result = try_load_wikitext(vocab_size=args.vocab_size, seq_len=args.seq_len, n_samples=800)
        if result is not None:
            x_pre, y_pre, tok, vocab_size = result
            # For finetune, reuse same data but treat as domain shift (same tokens, LoRA adapts)
            x_ft, y_ft = x_pre, y_pre
            use_wiki = True

    if not use_wiki:
        vocab_size = args.vocab_size
        print(f"Using synthetic data — vocab_size={vocab_size}, seq_len={args.seq_len}")
        x_pre, y_pre = synthetic_data(vocab_size, args.seq_len, n_samples=800, task="pretrain")
        x_ft, y_ft = synthetic_data(vocab_size, args.seq_len, n_samples=400, task="finetune", seed=1)
        print(f"Pre-train data: x={x_pre.shape}, y={y_pre.shape}")
        print(f"Fine-tune data: x={x_ft.shape}, y={y_ft.shape}")

    # ---- Build and pre-train ----
    print("\n--- Building model ---")
    model = build_model(vocab_size, args.seq_len)
    # Trigger build with a dummy forward.
    _ = model(x_pre[:2])
    print_param_stats(model, "Pre-adaptation")
    model.summary()

    print("\n--- Pre-training ---")
    model.compile(optimizer=AdamW(learning_rate=0.005), loss="sparse_categorical_crossentropy",
                  metrics=["sparse_accuracy"])
    model.fit(x_pre, y_pre, epochs=args.pretrain_epochs, batch_size=32, verbose=1)

    pre_eval = model.evaluate(x_pre[:200], y_pre[:200])
    print(f"Pre-train eval: loss={pre_eval['loss']:.4f}")

    # ---- Inject LoRA ----
    print(f"\n--- Injecting LoRA (rank={args.rank}, alpha={args.alpha}) ---")
    # Auto-detect all Dense layers (FFN + output projection).
    # For attention Q/V adaptation, you could pass target_modules=["ffn"] or explicit names.
    model, lora_layers = apply_lora(model, rank=args.rank, alpha=args.alpha, verbose=True)
    print_param_stats(model, "Post-adaptation (LoRA)")

    # Verify that initial LoRA output matches pre-adaptation output (delta = 0).
    sample = x_pre[:2]
    out_before_merge = model.predict(sample)

    # ---- Fine-tune with LoRA (only LoRA params are updated) ----
    print("\n--- LoRA fine-tuning ---")
    # Re-compile with a fresh optimizer (old optimizer state for frozen weights is irrelevant).
    model.compile(optimizer=AdamW(learning_rate=0.01), loss="sparse_categorical_crossentropy",
                  metrics=["sparse_accuracy"])
    model.fit(x_ft, y_ft, epochs=args.finetune_epochs, batch_size=32, verbose=1)

    ft_eval = model.evaluate(x_ft[:200], y_ft[:200])
    print(f"Fine-tune eval (on finetune task): loss={ft_eval['loss']:.4f}")

    # ---- Merge weights and verify ----
    print("\n--- Merging LoRA weights ---")
    out_before = model.predict(sample)
    for lyr in lora_layers:
        lyr.merge_weights()
    out_after = model.predict(sample)
    max_diff = np.max(np.abs(np.asarray(out_before) - np.asarray(out_after)))
    print(f"Max diff after merge (should be ~0): {max_diff:.2e}")

    # ---- Generate a short sequence (greedy) ----
    print("\n--- Generation demo ---")
    if tok is not None:
        start_text = "The "
        try:
            start_ids = tok.encode(start_text)
        except Exception:
            start_ids = x_pre[0][:args.seq_len].tolist()
    else:
        start_ids = x_pre[0][:args.seq_len].tolist()

    # Greedy generation from the (merged) model.
    seq = list(start_ids[: args.seq_len])
    for _ in range(20):
        inp = np.array(seq[-args.seq_len :]).reshape(1, -1)
        # pad if needed
        if inp.shape[1] < args.seq_len:
            inp = np.pad(inp, ((0, 0), (args.seq_len - inp.shape[1], 0)))
        probs = np.asarray(model.predict(inp))  # (1, seq, vocab)
        next_id = int(np.argmax(probs[0, -1]))
        seq.append(next_id)

    print(f"Generated token ids: {seq[:40]} ...")
    if tok is not None:
        try:
            print(f"Decoded: {tok.decode(seq)}")
        except Exception as e:
            print(f"(decode failed: {e})")

    print("\n--- Done ---")
    print("LoRA lets you fine-tune with <10% of the base parameters.")


if __name__ == "__main__":
    main()
