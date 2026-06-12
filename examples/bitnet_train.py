"""
BitNet Training Example
=======================
Trains a 1-bit / 1.58-bit BitNet language model on WikiText-2.

BitNet replaces all nn.Linear layers with BitLinear, which quantizes weights
to {-1, +1} (b1) or {-1, 0, +1} (b1.58) and activations to 8-bit.

Paper: https://arxiv.org/abs/2310.11453
"""

import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models import Sequential
from neutro.layers import Embedding, Dense
from neutro.layers.transformer.bitnet_block import BitNetBlock
from neutro.layers.normalization.rmsnorm import RMSNorm
from neutro.optimizers import AdamW
from neutro.utils.data_utils import load_wikitext2
from neutro.tokenizers import get_gpt2_tokenizer


def prepare_data(text, tokenizer, seq_len=64, step=3):
    encoded = tokenizer.encode(text)
    encoded = encoded[:50000]

    x, y = [], []
    for i in range(0, len(encoded) - seq_len, step):
        x.append(encoded[i:i + seq_len])
        y.append(encoded[i + 1:i + seq_len + 1])

    return np.array(x), np.array(y)


def build_bitnet_model(tokenizer, seq_len=32, dim=128, n_layers=2, n_heads=4, mode='b1.58'):
    model = Sequential([
        Embedding(tokenizer.vocab_size, dim, input_shape=(seq_len,)),
    ])

    for _ in range(n_layers):
        model.add(BitNetBlock(
            embed_dim=dim,
            num_heads=n_heads,
            ff_dim=dim * 2,
            mode=mode,
            activation_bits=8,
        ))

    model.add(RMSNorm())
    model.add(Dense(tokenizer.vocab_size))
    return model


def train_bitnet(mode='b1.58', epochs=2):
    print("Loading WikiText-2...")
    text = load_wikitext2()

    print("Loading GPT-2 tokenizer...")
    tokenizer = get_gpt2_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size (GPT-2): {vocab_size}")

    seq_len = 32
    print("Preparing data...")
    x, y = prepare_data(text, tokenizer, seq_len=seq_len)

    print(f"Building BitNet model (mode={mode})...")
    model = build_bitnet_model(tokenizer, seq_len=seq_len, dim=128, n_layers=2, n_heads=4, mode=mode)
    model.compile(
        optimizer=AdamW(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_accuracy'],
    )
    model.summary()

    print(f"Training (samples={len(x)}, epochs={epochs})...")
    x_sub = x[:500]
    y_sub = y[:500]
    history = model.fit(x_sub, y_sub, epochs=epochs, batch_size=16)

    print(f"\nFinal loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['sparse_accuracy'][-1]:.4f}")

    print("\nGenerating text...")
    start_text = "The "
    indices = tokenizer.encode(start_text)

    for _ in range(100):
        curr_x = indices[-seq_len:]
        if len(curr_x) < seq_len:
            curr_x = np.pad(curr_x, (seq_len - len(curr_x), 0))

        preds = model.predict(curr_x.reshape(1, -1))
        next_idx = np.argmax(preds[0, -1, :])
        indices = np.append(indices, next_idx)

    generated = tokenizer.decode(indices)
    print(f"Generated text: {generated}")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a BitNet language model")
    parser.add_argument("--mode", type=str, default="b1.58", choices=["b1", "b1.58"],
                        help="Quantization mode: b1 (binary) or b1.58 (ternary)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    args = parser.parse_args()

    train_bitnet(mode=args.mode, epochs=args.epochs)
