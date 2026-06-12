"""
Reasoning LLM Example
=====================
Trains a tiny causal transformer to do arithmetic reasoning.

All addition pairs are exactly 10 chars: Q:1+1=?A:2
No padding needed — every input is the same length.
The model sees Q:a+b=?A: and must predict the answer r.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models import Sequential
from neutro.layers import TokenPositionEmbedding, TransformerBlock, Dense
from neutro.optimizers import AdamW
from neutro.losses import SparseCategoricalCrossentropy


def generate_data():
    pairs = [(a, b, a + b) for a in range(1, 10) for b in range(1, 10) if a + b <= 9]
    return pairs


class CharTokenizer:
    def __init__(self, samples):
        chars = set()
        for s in samples:
            chars.update(s)
        chars = sorted(chars)
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
        self.idx_to_char[0] = ""
        self.vocab_size = len(chars) + 1

    def encode(self, text):
        return np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)

    def decode(self, ids):
        return "".join(self.idx_to_char.get(int(i), "") for i in ids)


def build_dataset(pairs):
    raw = [f"Q:{a}+{b}=?A:{r}" for a, b, r in pairs]
    tokenizer = CharTokenizer(raw)
    x = []
    y = []
    for s in raw:
        ids = tokenizer.encode(s)
        x.append(ids[:-1])
        y.append(ids[1:])
    return np.array(x), np.array(y), tokenizer


def build_model(vocab_size, seq_len, dim=64, n_layers=2, n_heads=4):
    return Sequential([
        TokenPositionEmbedding(vocab_size, max_len=seq_len, dim=dim, input_shape=(seq_len,)),
        TransformerBlock(embed_dim=dim, num_heads=n_heads, ff_dim=dim * 2,
                         causal=True, use_flash=True),
        TransformerBlock(embed_dim=dim, num_heads=n_heads, ff_dim=dim * 2,
                         causal=True, use_flash=True),
        Dense(vocab_size),
    ])


def main():
    all_pairs = generate_data()
    np.random.seed(42)
    np.random.shuffle(all_pairs)

    split = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]

    train_raw = [f"Q:{a}+{b}=?A:{r}" for a, b, r in train_pairs]
    test_raw = [f"Q:{a}+{b}=?A:{r}" for a, b, r in test_pairs]

    x_train, y_train, tokenizer = build_dataset(train_pairs)

    seq_len = x_train.shape[1]

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Characters: {list(tokenizer.char_to_idx.keys())}")
    print(f"Pairs: {len(all_pairs)} total, {len(train_pairs)} train, {len(test_pairs)} test")
    print(f"Sequence length (input): {seq_len}")
    print(f"Example: {train_raw[0]}")
    print(f"  x dims: {x_train.shape}, y dims: {y_train.shape}")

    model = build_model(tokenizer.vocab_size, seq_len)
    model.compile(
        optimizer=AdamW(learning_rate=0.002),
        loss=SparseCategoricalCrossentropy(from_logits=True),
    )
    model.summary()

    print("Training...")
    history = model.fit(x_train, y_train, epochs=300, batch_size=len(x_train), verbose=0)
    print(f"Final train loss: {history.history['loss'][-1]:.4f}")

    def evaluate(pairs, label):
        correct = 0
        for a, b, r in pairs:
            prompt = f"Q:{a}+{b}=?A:"
            expected = str(r)
            input_ids = tokenizer.encode(prompt).reshape(1, -1)
            preds = model.predict(input_ids)
            predicted_idx = int(np.argmax(preds[0, -1, :]))
            predicted_char = tokenizer.idx_to_char.get(predicted_idx, "?")
            match = "✓" if predicted_char == expected else "✗"
            print(f"  {prompt:<12} → {predicted_char} (expected: {expected})  {match}")
            if predicted_char == expected:
                correct += 1
        acc = f"{correct}/{len(pairs)} ({100*correct/len(pairs):.0f}%)"
        print(f"  {label}: {acc}")
        return acc

    print()
    evaluate(test_pairs, "Test accuracy")
    print()
    evaluate(train_pairs, "Train accuracy")


if __name__ == "__main__":
    main()
