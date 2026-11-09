import subprocess
import sys
from pathlib import Path

import numpy as np

from neutro.layers import Dense
from neutro.models import Sequential
from neutro.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]


def test_hello_example_runs():
    """The first example must stay runnable on a clean checkout."""
    result = subprocess.run(
        [sys.executable, "examples/00_hello_neutro.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Total params:" in result.stdout
    assert "Loss after epoch 5:" in result.stdout


def test_quickstart_training_loop():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4))
    y = np.eye(2)[rng.integers(0, 2, size=64)]

    model = Sequential([
        Dense(8, activation="relu", input_shape=(4,)),
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer=Adam(0.01), loss="categorical_crossentropy")

    history = model.fit(x, y, epochs=2, batch_size=16, verbose=0)
    assert len(history.history["loss"]) == 2
    assert np.isfinite(history.history["loss"][-1])

    predictions = model.predict(x[:3])
    assert predictions.shape == (3, 2)
    np.testing.assert_allclose(
        np.asarray(predictions.data).sum(axis=-1), 1.0, rtol=1e-6
    )
