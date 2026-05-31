import numpy as np
from neutro.layers.embedding.time_embedding import TimeEmbedding

def test_time_embedding():
    layer = TimeEmbedding(dim=128)
    t = np.array([0, 10, 50, 100])
    out = layer.forward(t)
    assert out.shape == (4, 128)
    # Check that t=0 has non-zero embedding (it's sinusoidal)
    assert np.any(out[0] != 0)
    
    # Check that different t give different embeddings
    out10 = layer.forward(np.array([10]))
    out50 = layer.forward(np.array([50]))
    assert not np.allclose(out10, out50)

def test_time_embedding_backward_shape():
    layer = TimeEmbedding(dim=64)
    t = np.array([1, 2, 3])
    _ = layer.forward(t)
    grad = layer.backward(np.ones((3, 64)))

    assert grad.shape == t.shape
    assert np.all(grad == 0)

def test_time_embedding_build():
    layer = TimeEmbedding(dim=128)
    layer.build((4,))
    assert layer.built

def test_time_embedding_compute_output_shape():
    layer = TimeEmbedding(dim=256)
    shape = layer.compute_output_shape((4,))
    assert shape == (4, 256)

def test_time_embedding_2d_input():
    layer = TimeEmbedding(dim=64)
    t = np.array([[0], [10], [50], [100]])
    out = layer.forward(t)
    assert out.shape == (4, 64)

def test_time_embedding_odd_dim():
    layer = TimeEmbedding(dim=129)
    t = np.array([0, 10, 50])
    out = layer.forward(t)
    assert out.shape == (3, 129)
