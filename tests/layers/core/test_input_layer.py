import numpy as np
import pytest
from neutro.layers.core.input_layer import InputLayer, Input
from neutro.engine.node import KerasTensor


def test_input_layer_forward():
    layer = InputLayer(input_shape=(4,))
    out = layer.forward(np.array([1, 2, 3, 4]))
    assert np.array_equal(out, np.array([1, 2, 3, 4]))


def test_input_layer_backward():
    layer = InputLayer(input_shape=(4,))
    grad = layer.backward(np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.array_equal(grad, np.array([0.1, 0.2, 0.3, 0.4]))


def test_input_layer_build_immediate():
    layer = InputLayer(input_shape=(28, 28, 1))
    assert layer.built
    assert layer.input_shape == (28, 28, 1)


def test_input_layer_build_explicit():
    layer = InputLayer()
    layer.build((None, 28, 28, 1))
    assert layer.built
    assert layer.input_shape == (None, 28, 28, 1)


def test_input_no_shape_raises():
    with pytest.raises(ValueError, match="Please provide a shape"):
        Input(shape=None)


def test_input_with_list_shape():
    tensor = Input(shape=[28, 28, 1])
    assert isinstance(tensor, KerasTensor)
    assert tensor.shape == (None, 28, 28, 1)


def test_input_with_tuple_shape():
    tensor = Input(shape=(28, 28, 1))
    assert isinstance(tensor, KerasTensor)
    assert tensor.shape == (None, 28, 28, 1)


def test_input_with_batch_shape():
    tensor = Input(shape=(None, 28, 28, 1))
    assert isinstance(tensor, KerasTensor)
    assert tensor.shape == (None, 28, 28, 1)
