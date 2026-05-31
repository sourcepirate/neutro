import numpy as np
from neutro.models import Model, Sequential
from neutro.layers import Dense, Input
from neutro.layers.base import Layer


class ContainerLayer(Layer):
    def __init__(self, units=5):
        super().__init__()
        self.dense = Dense(units)

    def forward(self, x, training=False):
        return self.dense(x, training=training)

    def backward(self, grad_output):
        return self.dense.backward(grad_output)


class DoubleRefLayer(Layer):
    def __init__(self, units=5):
        super().__init__()
        self.inner = Dense(units)
        self.inner_copy = self.inner

    def forward(self, x, training=False):
        return self.inner(x, training=training)

    def backward(self, grad_output):
        return self.inner.backward(grad_output)


class SubclassNoBuild(Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(10)

    def forward(self, x, training=False):
        return self.dense(x, training=training)


class BrokenLayer(Layer):
    def compute_output_shape(self, input_shape):
        raise ValueError("broken")


def test_clear_layer_grads_with_sublayers():
    model = Sequential([ContainerLayer(4), Dense(3)])
    model.build((None, 4))
    x = np.random.rand(2, 4)
    out = model.forward(x, training=True)
    grad = np.random.rand(2, 3)
    model.backward(grad)
    container = model.layers[0]
    assert len(container.dense.grads) > 0
    Model._clear_layer_grads(model)
    assert len(container.dense.grads) == 0


def test_accumulate_layer_grads_visited_check():
    layer = DoubleRefLayer(5)
    layer.build((None, 4))
    layer.inner.build((None, 4))
    layer.grads['W'] = np.ones((4, 5))
    layer.inner.grads['W'] = np.ones((4, 5)) * 2
    accumulator = {}
    Model._accumulate_layer_grads(layer, accumulator)
    double_ref_id = id(layer)
    inner_id = id(layer.inner)
    assert double_ref_id in accumulator
    assert inner_id in accumulator
    assert np.all(accumulator[double_ref_id]['W'] == 1.0)
    assert np.all(accumulator[inner_id]['W'] == 2.0)


def test_restore_layer_state_with_sublayers():
    container = ContainerLayer(5)
    container.build((None, 4))
    container.dense.build((None, 4))
    container.dense.custom_attr = "original"
    state = Model._capture_layer_state(container)
    container.dense.custom_attr = "modified"
    assert container.dense.custom_attr == "modified"
    Model._restore_layer_state(container, state)
    assert container.dense.custom_attr == "original"


def test_functional_compute_output_shape():
    inputs = Input(shape=(10,))
    x = Dense(5)(inputs)
    outputs = Dense(3)(x)
    model = Model(inputs=inputs, outputs=outputs)
    shape = model.compute_output_shape((None, 10))
    assert shape == (None, 3)


def test_functional_build():
    inputs = Input(shape=(10,))
    x = Dense(5)(inputs)
    outputs = Dense(3)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.build((None, 10))
    assert model.built is True


def test_backward_functional_single_output():
    inputs = Input(shape=(10,))
    x = Dense(5, activation='relu')(inputs)
    outputs = Dense(3)(x)
    model = Model(inputs=inputs, outputs=outputs)
    x_data = np.random.rand(4, 10)
    y = model.forward(x_data, training=True)
    grad = np.random.rand(4, 3)
    grad_inputs = model.backward(grad)
    assert grad_inputs.shape == (4, 10)


def test_subclassed_model_build_no_override():
    model = SubclassNoBuild()
    model.build((None, 5))
    assert model.built is True
    assert model.input_shape == (None, 5)


def test_sequential_forward_without_kv_cache():
    model = Sequential([Dense(5, input_shape=(10,)), Dense(3)])
    x = np.random.rand(4, 10)
    out = model.forward(x, training=False, kv_cache=None)
    assert out.shape == (4, 3)


def test_sequential_add_with_input_shape():
    model = Sequential()
    dense = Dense(5, input_shape=(10,))
    model.add(dense)
    assert len(model.layers) == 1
    assert model.layers[0].built


def test_summary_unbuilt_layer(capsys):
    model = Sequential()
    layer = Dense(5)
    model.layers.append(layer)
    model.summary()
    captured = capsys.readouterr()
    assert "unbuilt" in captured.out


def test_summary_exception_built(capsys):
    model = Sequential()
    layer = BrokenLayer()
    layer.built = True
    layer.input_shape = (None, 10)
    model.layers.append(layer)
    model.summary()
    captured = capsys.readouterr()
    assert "multiple" in captured.out
