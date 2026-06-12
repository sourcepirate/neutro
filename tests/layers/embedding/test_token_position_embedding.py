import numpy as np
from neutro.layers.embedding.token_position_embedding import TokenPositionEmbedding


def test_forward_shape():
    layer = TokenPositionEmbedding(vocab_size=16, max_len=9, dim=64)
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    out = layer(x, training=True)
    assert out.shape == (1, 9, 64)


def test_batch_forward():
    layer = TokenPositionEmbedding(vocab_size=16, max_len=9, dim=64)
    x = np.random.randint(0, 16, (8, 9))
    out = layer(x, training=True)
    assert out.shape == (8, 9, 64)


def test_backward():
    layer = TokenPositionEmbedding(vocab_size=16, max_len=9, dim=64)
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    out = layer(x, training=True)
    grad_output = np.random.randn(1, 9, 64)
    grad = layer.backward(grad_output)
    assert grad is None
    assert layer.token_emb.grads['embeddings'].shape == (16, 64)
    assert layer.pos_emb.grads['embeddings'].shape == (9, 64)
    assert not np.all(layer.token_emb.grads['embeddings'] == 0)
    assert not np.all(layer.pos_emb.grads['embeddings'] == 0)


def test_backward_batch():
    layer = TokenPositionEmbedding(vocab_size=16, max_len=9, dim=64)
    x = np.random.randint(0, 16, (8, 9))
    out = layer(x, training=True)
    grad_output = np.random.randn(8, 9, 64)
    grad = layer.backward(grad_output)
    assert grad is None
    assert layer.token_emb.grads['embeddings'].shape == (16, 64)
    assert layer.pos_emb.grads['embeddings'].shape == (9, 64)


def test_compute_output_shape():
    layer = TokenPositionEmbedding(vocab_size=16, max_len=9, dim=64)
    shape = layer.compute_output_shape((None, 9))
    assert shape == (None, 9, 64)


def test_sublayers():
    layer = TokenPositionEmbedding(vocab_size=16, max_len=9, dim=64)
    sub = layer.sublayers
    assert len(sub) == 2
    assert layer.token_emb in sub
    assert layer.pos_emb in sub
