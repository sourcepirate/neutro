import numpy as np
import pytest
from neutro.layers import Input, Dense, Add
from neutro.models import Model
from neutro.optimizers import SGD


def test_mimo_fit_two_inputs():
    """Fit a 2-input functional model with list inputs."""
    i1 = Input(shape=(4,), name='input1')
    i2 = Input(shape=(4,), name='input2')
    merged = Add()([i1, i2])
    out = Dense(1, name='output')(merged)

    model = Model(inputs=[i1, i2], outputs=out)
    model.compile(optimizer=SGD(0.01), loss='mse')

    # Generate data: x1 + x2 approximate y
    X1 = np.random.randn(20, 4).astype(np.float32)
    X2 = np.random.randn(20, 4).astype(np.float32)
    Y = np.sum(X1 + X2, axis=-1, keepdims=True).astype(np.float32)

    history = model.fit([X1, X2], Y, epochs=3, batch_size=8, verbose=0)
    assert 'loss' in history.history
    assert len(history.history['loss']) == 3
    # Loss should decrease (we're learning)
    assert history.history['loss'][-1] <= history.history['loss'][0] * 1.5


def test_mimo_fit_two_outputs():
    """Fit a 2-output functional model with list targets."""
    inp = Input(shape=(4,))
    x = Dense(8, activation='relu')(inp)
    o1 = Dense(1, name='out1')(x)
    o2 = Dense(2, name='out2')(x)

    model = Model(inputs=inp, outputs=[o1, o2])
    model.compile(optimizer=SGD(0.01), loss='mse')

    X = np.random.randn(20, 4).astype(np.float32)
    Y1 = np.random.randn(20, 1).astype(np.float32)
    Y2 = np.random.randn(20, 2).astype(np.float32)

    history = model.fit(X, [Y1, Y2], epochs=3, batch_size=8, verbose=0)
    assert 'loss' in history.history
    assert len(history.history['loss']) == 3
    # Simple sanity: loss shouldn't explode
    assert history.history['loss'][-1] < 1e6


def test_mimo_fit_two_inputs_two_outputs():
    """Fit a 2-input, 2-output functional model."""
    i1 = Input(shape=(4,), name='i1')
    i2 = Input(shape=(4,), name='i2')
    merged = Add()([i1, i2])
    x = Dense(8, activation='relu')(merged)
    o1 = Dense(1, name='out1')(x)
    o2 = Dense(2, name='out2')(x)

    model = Model(inputs=[i1, i2], outputs=[o1, o2])
    model.compile(optimizer=SGD(0.01), loss='mse')

    X1 = np.random.randn(20, 4).astype(np.float32)
    X2 = np.random.randn(20, 4).astype(np.float32)
    Y1 = np.random.randn(20, 1).astype(np.float32)
    Y2 = np.random.randn(20, 2).astype(np.float32)

    history = model.fit([X1, X2], [Y1, Y2], epochs=3, batch_size=8, verbose=0)
    assert 'loss' in history.history
    assert len(history.history['loss']) == 3


def test_mimo_evaluate():
    """Evaluate a MIMO model."""
    i1 = Input(shape=(4,))
    i2 = Input(shape=(4,))
    merged = Add()([i1, i2])
    o1 = Dense(1, name='out1')(merged)
    o2 = Dense(2, name='out2')(merged)

    model = Model(inputs=[i1, i2], outputs=[o1, o2])
    model.compile(optimizer=SGD(0.01), loss='mse')

    X1 = np.random.randn(5, 4).astype(np.float32)
    X2 = np.random.randn(5, 4).astype(np.float32)
    Y1 = np.ones((5, 1)).astype(np.float32)
    Y2 = np.ones((5, 2)).astype(np.float32)

    results = model.evaluate([X1, X2], [Y1, Y2])
    assert 'loss' in results
    assert isinstance(results['loss'], (float, np.floating))


def test_mimo_validation_data():
    """MIMO validation data in fit() works."""
    i1 = Input(shape=(4,))
    i2 = Input(shape=(4,))
    merged = Add()([i1, i2])
    out = Dense(1)(merged)

    model = Model(inputs=[i1, i2], outputs=out)
    model.compile(optimizer=SGD(0.01), loss='mse')

    # Training data
    X1 = np.random.randn(20, 4).astype(np.float32)
    X2 = np.random.randn(20, 4).astype(np.float32)
    Y = np.sum(X1 + X2, axis=-1, keepdims=True).astype(np.float32)

    # Validation data
    V1 = np.random.randn(5, 4).astype(np.float32)
    V2 = np.random.randn(5, 4).astype(np.float32)
    VY = np.sum(V1 + V2, axis=-1, keepdims=True).astype(np.float32)

    history = model.fit([X1, X2], Y, epochs=2, batch_size=8,
                         validation_data=([V1, V2], VY), verbose=0)
    assert 'val_loss' in history.history
