import numpy as np
import pytest
from neutro.layers.core.dropout import Dropout
from neutro.models.base_model import Sequential


def test_dropout_inference():
    layer = Dropout(0.5)
    x = np.random.rand(10, 10)

    out_inf = layer.forward(x, training=False)
    assert np.all(out_inf == x)


def test_dropout_training():
    layer = Dropout(0.5)
    x = np.random.rand(10, 10)

    out_train = layer.forward(x, training=True)
    assert not np.all(out_train == x)


def test_dropout_rate_zero():
    layer = Dropout(0.0)
    x = np.random.rand(10, 10)

    out = layer.forward(x, training=True)
    assert np.all(out == x)

    grad = np.random.rand(10, 10)
    dx = layer.backward(grad)
    assert np.all(dx == grad)


def test_dropout_1d_input():
    layer = Dropout(0.5)
    x = np.random.rand(20)

    out = layer.forward(x, training=True)
    assert out.shape == (20,)
    assert not np.all(out == x)

    grad = np.random.rand(20)
    dx = layer.backward(grad)
    assert dx.shape == (20,)


def test_dropout_3d_input():
    layer = Dropout(0.3)
    x = np.random.rand(4, 16, 64)

    out = layer.forward(x, training=True)
    assert out.shape == (4, 16, 64)

    grad = np.random.rand(4, 16, 64)
    dx = layer.backward(grad)
    assert dx.shape == (4, 16, 64)


def test_dropout_statistics():
    layer = Dropout(0.5)
    x = np.ones((1000, 100))

    out = layer.forward(x, training=True)
    zero_fraction = np.mean(out == 0)
    assert 0.45 < zero_fraction < 0.55


def test_dropout_backward_inference():
    layer = Dropout(0.5)
    x = np.random.rand(10, 10)
    grad = np.random.rand(10, 10)

    layer.forward(x, training=True)
    layer.forward(x, training=False)
    dx = layer.backward(grad)
    assert np.all(dx == grad)


def test_dropout_backward_values():
    layer = Dropout(0.5)
    x = np.ones((10, 10))
    grad = np.ones((10, 10))

    layer.forward(x, training=True)

    dx = layer.backward(grad)
    expected_dx = grad * layer.mask
    np.testing.assert_allclose(dx, expected_dx)


def test_dropout_backward_no_forward():
    layer = Dropout(0.5)

    grad = np.random.rand(10, 10)
    dx = layer.backward(grad)
    assert np.all(dx == grad)


def test_dropout_compute_output_shape():
    layer = Dropout(0.5)

    shape = layer.compute_output_shape((None, 32))
    assert shape == (None, 32)

    shape = layer.compute_output_shape((16, 32))
    assert shape == (16, 32)

    shape = layer.compute_output_shape((None, 16, 64))
    assert shape == (None, 16, 64)


def test_dropout_in_sequential_model():
    model = Sequential([
        Dropout(0.5),
        Dropout(0.3),
        Dropout(0.0),
    ])
    x = np.random.rand(8, 32)

    out = model.forward(x, training=True)
    assert out.shape == (8, 32)

    out_inf = model.forward(x, training=False)
    assert np.all(out_inf == x)


def test_dropout_mask_recreated_each_forward():
    layer = Dropout(0.5)
    x = np.ones((100, 100))

    out1 = layer.forward(x, training=True)
    mask1 = (out1 != 0).astype(float)
    out2 = layer.forward(x, training=True)
    mask2 = (out2 != 0).astype(float)

    assert not np.all(mask1 == mask2)
