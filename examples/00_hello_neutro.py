"""Your first Neutro model — no downloads, no GPU, about two seconds.

Run this file from the repository root::

    python examples/00_hello_neutro.py

It builds a tiny two-layer network, trains it on synthetic data, and prints
the loss.  The goal is to show the whole loop in one screen:

    data -> model -> loss -> backward -> optimizer step

Read `docs/tutorials/01-first-model.md` for the line-by-line version.
"""

import numpy as np

from neutro.layers import Dense
from neutro.models import Sequential
from neutro.optimizers import Adam


def make_data(n_samples=256, n_features=8, n_classes=3, seed=0):
    """Create an easy, deterministic classification problem."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float64)

    # A simple linear rule plus a little noise.  The model only has to learn
    # which of the three class-score directions is largest.
    scores = x @ rng.normal(size=(n_features, n_classes))
    labels = np.argmax(scores, axis=1)

    # Keras-style one-hot targets: (n_samples, n_classes)
    y = np.eye(n_classes)[labels]
    return x, y


def main():
    x, y = make_data()

    model = Sequential([
        Dense(16, activation="relu", input_shape=(x.shape[1],), name="hidden"),
        Dense(y.shape[1], activation="softmax", name="scores"),
    ])

    # Both Keras-style strings and optimizer objects work.
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Input shape: ", x.shape, " (batch, features)")
    print("Target shape:", y.shape, " (batch, classes)")
    print()
    print("Model summary")
    model.summary()

    history = model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    losses = history.history["loss"]

    print()
    print(f"Loss after epoch 1: {losses[0]:.4f}")
    print(f"Loss after epoch 5: {losses[-1]:.4f}")
    print()
    print("The loss should go down.  That is the whole training loop:")
    print("  forward pass -> loss -> backward pass -> optimizer step")
    print()
    print("Next: open docs/tutorials/01-first-model.md")


if __name__ == "__main__":
    main()
