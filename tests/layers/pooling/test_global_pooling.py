import numpy as np
import pytest
from neutro.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

def test_global_average_pooling():
    batch, h, w, c = 2, 4, 4, 3
    layer = GlobalAveragePooling2D()
    
    inputs = np.random.randn(batch, h, w, c)
    out = layer.forward(inputs)
    
    assert out.shape == (batch, c)
    assert np.allclose(out[0, 0], np.mean(inputs[0, :, :, 0]))

def test_global_max_pooling():
    batch, h, w, c = 2, 4, 4, 3
    layer = GlobalMaxPooling2D()
    
    inputs = np.random.randn(batch, h, w, c)
    out = layer.forward(inputs)
    
    assert out.shape == (batch, c)
    assert out[0, 0] == np.max(inputs[0, :, :, 0])

def test_global_pooling_backward():
    batch, h, w, c = 2, 4, 4, 3
    avg_layer = GlobalAveragePooling2D()
    max_layer = GlobalMaxPooling2D()
    
    inputs = np.random.randn(batch, h, w, c)
    
    # Avg
    avg_layer.forward(inputs)
    grad_avg = avg_layer.backward(np.random.randn(batch, c))
    assert grad_avg.shape == inputs.shape
    
    # Max
    max_layer.forward(inputs)
    grad_max = max_layer.backward(np.random.randn(batch, c))
    assert grad_max.shape == inputs.shape

def test_global_pooling_channels_first():
    inputs = np.random.randn(2, 3, 4, 4)

    avg_layer = GlobalAveragePooling2D(data_format='channels_first')
    avg_out = avg_layer.forward(inputs)
    assert avg_out.shape == (2, 3)
    assert np.allclose(avg_out[0, 0], np.mean(inputs[0, 0, :, :]))
    assert avg_layer.backward(np.random.randn(2, 3)).shape == inputs.shape

    max_layer = GlobalMaxPooling2D(data_format='channels_first')
    max_out = max_layer.forward(inputs)
    assert max_out.shape == (2, 3)
    assert max_out[0, 0] == np.max(inputs[0, 0, :, :])
    assert max_layer.backward(np.random.randn(2, 3)).shape == inputs.shape

def test_global_avg_pooling_invalid_data_format():
    with pytest.raises(ValueError, match="data_format must be"):
        GlobalAveragePooling2D(data_format='invalid')

def test_global_max_pooling_invalid_data_format():
    with pytest.raises(ValueError, match="data_format must be"):
        GlobalMaxPooling2D(data_format='invalid')

def test_global_avg_pooling_compute_output_shape():
    layer = GlobalAveragePooling2D(data_format='channels_last')
    shape = layer.compute_output_shape((2, 8, 8, 3))
    assert shape == (2, 3)

def test_global_max_pooling_compute_output_shape():
    layer = GlobalMaxPooling2D(data_format='channels_last')
    shape = layer.compute_output_shape((2, 8, 8, 3))
    assert shape == (2, 3)
