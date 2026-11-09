"""Qwen dense vs Qwen-MoE toy comparison.

Same arithmetic reasoning task as reasoning_train.py, tiny CPU config.
Prints train loss + router utilization for the MoE model.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models.language.qwen import Qwen2Tiny, Qwen3MoETiny
from neutro.optimizers import AdamW
from neutro.losses import SparseCategoricalCrossentropy


class CharTokenizer:
    def __init__(self, samples):
        chars = sorted({c for s in samples for c in s})
        self.c2i = {ch: i + 1 for i, ch in enumerate(chars)}
        self.i2c = {i + 1: ch for i, ch in enumerate(chars)}
        self.i2c[0] = ""
        self.vocab_size = len(chars) + 1

    def encode(self, text):
        return np.array([self.c2i[c] for c in text], dtype=np.int32)


def build_dataset():
    pairs = [(a, b, a + b) for a in range(1, 10) for b in range(1, 10) if a + b <= 9]
    raw = [f"Q:{a}+{b}=?A:{r}" for a, b, r in pairs]
    tok = CharTokenizer(raw)
    x = np.stack([tok.encode(s[:-1]) for s in raw])
    y = np.stack([tok.encode(s[1:]) for s in raw])
    return x, y, tok


def run(name, model, x, y):
    model.compile(optimizer=AdamW(learning_rate=0.002),
                  loss=SparseCategoricalCrossentropy(from_logits=True))
    h = model.fit(x, y, epochs=5, batch_size=len(x), verbose=0)
    print(f"{name}: final loss {h.history['loss'][-1]:.4f}")
    return model


def router_report(model):
    for layer in model.layers:
        for sub in getattr(layer, 'sublayers', []):
            if type(sub).__name__ == 'MoELayer' and hasattr(sub, '_last_frac'):
                print(f"  {layer.name or type(layer).__name__}: "
                      f"routed_frac={np.round(sub._last_frac, 3)} "
                      f"mean_prob={np.round(sub._last_mean_prob, 3)} "
                      f"aux={sub.aux_loss():.4f}")


def main():
    x, y, tok = build_dataset()
    seq_len = x.shape[1]
    print(f"vocab={tok.vocab_size} seq={seq_len} n={len(x)}")

    dense = Qwen2Tiny(tok.vocab_size, seq_len, dim=32, n_layers=1, n_heads=4)
    run("Qwen2-dense", dense, x, y)

    moe = Qwen3MoETiny(tok.vocab_size, seq_len, dim=32, n_layers=1, n_heads=4,
                       n_experts=8, top_k=2)
    run("Qwen3-MoE", moe, x, y)
    router_report(moe)


if __name__ == "__main__":
    main()
