import numpy as np
import inspect
from neutro.models import Model, Sequential
from neutro.layers import Dense, Input, ReLU, Dropout
from neutro.layers.base import Layer
from neutro.layers.transformer.transformer_block import TransformerBlock
from neutro.layers.attention.kv_cache import KVCache
from neutro.optimizers import SGD


class SubclassedModel(Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense1 = Dense(8)
        self.relu = ReLU()
        self.dense2 = Dense(4)

    def forward(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.relu(x)
        return self.dense2(x)

    def build(self, input_shape):
        self.dense1.build(input_shape)
        shape = self.dense1.compute_output_shape(input_shape)
        self.relu.build(shape)
        shape = self.relu.compute_output_shape(shape)
        self.dense2.build(shape)
        self.built = True


# 1. _init_graph with single output (not a list) — lines 48, 55
def test_init_graph_single_output():
    inputs = Input(shape=(10,))
    x = Dense(5, activation='relu')(inputs)
    outputs = Dense(3)(x)
    model = Model(inputs=inputs, outputs=outputs)
    assert len(model._nodes_ordered) > 0
    assert len(model.layers) > 0


# 2. _get_all_layers without arguments — lines 72-73
def test_get_all_layers_no_args():
    model = Sequential([Dense(10), Dense(5)])
    all_layers = model._get_all_layers()
    assert len(all_layers) == 2


# 3. evaluate with metrics — lines 531-536
def test_evaluate_with_metrics():
    model = Sequential([Dense(8, input_shape=(4,)), Dense(2)])
    model.compile(optimizer=SGD(0.01), loss='mse', metrics=['accuracy'])
    x = np.random.rand(10, 4)
    y = np.random.rand(10, 2)
    model.fit(x, y, epochs=1, batch_size=4, verbose=0)
    results = model.evaluate(x, y)
    assert 'loss' in results
    assert 'accuracy' in results


# 4. fit with validation_data — validation loss/metrics path
def test_fit_with_validation_data():
    model = Sequential([Dense(8, input_shape=(4,)), Dense(2)])
    model.compile(optimizer=SGD(0.01), loss='mse', metrics=['accuracy'])
    x = np.random.rand(10, 4)
    y = np.random.rand(10, 2)
    val_data = (np.random.rand(5, 4), np.random.rand(5, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, verbose=0, validation_data=val_data)
    assert 'val_loss' in history.history
    assert 'val_accuracy' in history.history


# 5. backward functional path with single input/output — lines 426, 440, 465-470
def test_backward_functional_single_io():
    inputs = Input(shape=(4,))
    x = Dense(8, activation='relu')(inputs)
    outputs = Dense(2)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=SGD(0.01), loss='mse')
    x_data = np.random.rand(10, 4)
    y_data = np.random.rand(10, 2)
    model.fit(x_data, y_data, epochs=1, batch_size=4, verbose=0)
    for layer in model.layers:
        if layer.grads:
            for k, v in layer.grads.items():
                assert not np.allclose(v, 0)


# 6. build for subclassed model — lines 507-510
def test_build_subclassed_model():
    model = SubclassedModel()
    assert not model.built
    model.build((None, 6))
    assert model.built
    x = np.random.rand(5, 6)
    y = model.forward(x)
    assert y.shape == (5, 4)


# 7. summary on functional model — "Connected to" column
def test_summary_functional_model(capsys):
    inputs = Input(shape=(10,))
    x = Dense(5, activation='relu')(inputs)
    outputs = Dense(3)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    captured = capsys.readouterr()
    assert 'Connected to' in captured.out
    assert 'Total params' in captured.out


# 8. clear_layer_grads — static method clears recursively
def test_clear_layer_grads():
    model = Sequential([Dense(5, input_shape=(3,)), Dense(2)])
    model.build((None, 3))
    for layer in model.layers:
        for k in layer.params:
            layer.grads[k] = np.random.randn(*layer.params[k].shape)
    Model._clear_layer_grads(model)
    for layer in model.layers:
        assert layer.grads == {}


# 9a. _accumulate_layer_grads — new key path (line 157)
def test_accumulate_layer_grads_new_key():
    layer = Dense(5, input_shape=(3,))
    layer.build((None, 3))
    layer.grads['W'] = np.ones(layer.params['W'].shape)
    layer.grads['b'] = np.ones(layer.params['b'].shape)
    accumulator = {}
    Model._accumulate_layer_grads(layer, accumulator)
    l_id = id(layer)
    assert l_id in accumulator
    assert np.all(accumulator[l_id]['W'] == 1.0)
    assert np.all(accumulator[l_id]['b'] == 1.0)


# 9b. _accumulate_layer_grads — existing key path (line 154-155)
def test_accumulate_layer_grads_existing_key():
    layer = Dense(5, input_shape=(3,))
    layer.build((None, 3))

    layer.grads['W'] = np.ones(layer.params['W'].shape) * 2
    layer.grads['b'] = np.ones(layer.params['b'].shape) * 2

    accumulator = {}
    l_id = id(layer)
    accumulator[l_id] = {
        'W': np.ones(layer.params['W'].shape),
        'b': np.ones(layer.params['b'].shape)
    }

    Model._accumulate_layer_grads(layer, accumulator)

    assert np.all(accumulator[l_id]['W'] == 3.0)
    assert np.all(accumulator[l_id]['b'] == 3.0)


# 10. forward sequential with kv_cache — lines 370-377
def test_forward_sequential_with_kv_cache():
    block = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16, use_flash=True, causal=True, pre_norm=True)
    model = Sequential([block])
    x = np.random.rand(2, 4, 8)
    cache = KVCache()
    output = model.forward(x, training=False, kv_cache=cache)
    assert output.shape == (2, 4, 8)
