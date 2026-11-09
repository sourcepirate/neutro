import numpy as np
import pytest

from neutro.layers import Dense
from neutro.models import Sequential
from neutro.optimizers import Adam, AdamW, SGD


def _tiny_model(optimizer):
    model = Sequential([Dense(4, activation="relu", input_shape=(3,)), Dense(2)])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    return model


def test_compile_accepts_optimizer_names():
    assert isinstance(_tiny_model("sgd").optimizer, SGD)
    assert isinstance(_tiny_model("adam").optimizer, Adam)
    assert isinstance(_tiny_model("adamw").optimizer, AdamW)


def test_compile_accepts_optimizer_instances():
    original = Adam(learning_rate=0.123)
    assert _tiny_model(original).optimizer is original


def test_compile_rejects_unknown_optimizer_name():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        _tiny_model("definitely-not-an-optimizer")


def test_named_optimizer_can_train():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3))
    labels = rng.integers(0, 2, size=16)
    y = np.eye(2)[labels]

    model = _tiny_model("adam")
    history = model.fit(x, y, epochs=1, batch_size=8, verbose=0)
    assert np.isfinite(history.history["loss"][0])
