"""A guided tour of attention — shapes, softmax, heads, and causal masks.

Run from the repository root::

    python examples/03_attention_tour.py

It starts with a hand-written NumPy attention calculation, then uses Neutro's
MultiHeadAttention and a causal mask.
"""

import numpy as np

from neutro.layers import MultiHeadAttention
from neutro.layers.attention.base_attention import BaseAttention


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def main():
    rng = np.random.default_rng(0)

    # One "head" of attention by hand.
    batch, seq, dim, head_dim = 2, 4, 8, 4
    x = rng.normal(size=(batch, seq, dim))

    Wq = rng.normal(size=(dim, dim))
    Wk = rng.normal(size=(dim, dim))
    Wv = rng.normal(size=(dim, dim))

    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(head_dim)
    weights = softmax(scores)
    output = weights @ V

    print("Hand-written attention")
    print("  x:       ", x.shape, "(batch, seq, dim)")
    print("  Q/K/V:   ", Q.shape)
    print("  scores:  ", scores.shape, "(batch, query, key)")
    print("  weights: ", weights.shape)
    print("  output:  ", output.shape)
    print("  first query's weights sum to", round(float(weights[0, 0].sum()), 6))
    print()

    # The same idea, packaged as a multi-head layer.
    attention = MultiHeadAttention(num_heads=2, key_dim=8)
    y = attention(x)

    print("Neutro MultiHeadAttention")
    print("  output:  ", y.shape, type(y).__name__)
    print()

    # Causal mask: position i may only look at positions <= i.
    mask = BaseAttention.create_causal_mask(seq)
    print("Causal mask (1 = blocked)")
    print(mask)
    print()

    y_causal = attention(x, mask=mask)
    first_token_only = attention(x[:, :1, :])

    print("Causal output:              ", y_causal.shape)
    print(
        "First token unchanged:      ",
        np.allclose(y_causal.data[:, 0], first_token_only.data[:, 0], atol=1e-6),
    )
    print()
    print("That is attention: project -> score -> softmax -> weighted sum.")
    print("Next: docs/tutorials/03-attention-from-scratch.md")


if __name__ == "__main__":
    main()
