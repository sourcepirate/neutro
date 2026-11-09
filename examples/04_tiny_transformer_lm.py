"""A tiny causal transformer language model.

Run from the repository root::

    python examples/04_tiny_transformer_lm.py

The task is synthetic and exact: the next token is always ``(token + 1) % V``.
That makes it easy to see whether the model learned the pattern.
"""

import numpy as np

from neutro.layers import Dense, Embedding, TransformerBlock
from neutro.losses import SparseCategoricalCrossentropy
from neutro.models import Sequential
from neutro.optimizers import Adam

VOCAB_SIZE = 16
SEQ_LEN = 12


def make_data(n_samples=256, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, VOCAB_SIZE, size=(n_samples, SEQ_LEN))
    y = (x + 1) % VOCAB_SIZE
    return x, y


def main():
    x, y = make_data()

    model = Sequential([
        Embedding(VOCAB_SIZE, 32, input_shape=(SEQ_LEN,)),
        TransformerBlock(
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            causal=True,
            pre_norm=True,
        ),
        Dense(VOCAB_SIZE),   # raw logits; generate() applies softmax
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=SparseCategoricalCrossentropy(from_logits=True),
    )

    history = model.fit(x, y, epochs=10, batch_size=32, verbose=0)
    losses = np.round(history.history["loss"], 4)

    print("Training loss (first, middle, last):")
    print(" ", losses[0], losses[len(losses) // 2], losses[-1])
    print()

    logits = model.predict(x[:2])
    predictions = np.argmax(logits.data, axis=-1)

    print("Input:     ", x[0][:10])
    print("Target:    ", y[0][:10])
    print("Prediction:", predictions[0][:10])
    print()

    prompt = x[:1, :4]
    generated = model.generate(prompt, max_new_tokens=5)
    print("Prompt:   ", prompt)
    print("Generated:", generated)
    print()
    print("The model should continue the +1 sequence. Next: docs/tutorials/04-tiny-transformer-lm.md")


if __name__ == "__main__":
    main()
