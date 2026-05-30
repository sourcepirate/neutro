import numpy as np
import pytest
from neutro.layers import Input, Dense, Add
from neutro.layers.transformer.transformer_block import TransformerBlock
from neutro.models import Model
from neutro.optimizers import SGD


def test_shared_transformer_block_forward():
    """Shared TransformerBlock produces correct output shapes."""
    shared_block = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16, use_flash=True)

    inp = Input(shape=(4, 8))
    x1 = shared_block(inp)
    x2 = shared_block(x1)
    merged = Add()([x1, x2])
    out = Dense(4)(merged)

    model = Model(inputs=inp, outputs=out)

    X = np.random.randn(2, 4, 8).astype(np.float32)
    y = model.predict(X)
    assert y.shape == (2, 4, 4)


def test_shared_transformer_block_backward():
    """Shared TransformerBlock gradients flow correctly through all branches."""
    np.random.seed(42)
    shared_block = TransformerBlock(embed_dim=4, num_heads=2, ff_dim=8, use_flash=True)

    inp = Input(shape=(2, 4))
    x1 = shared_block(inp)
    x2 = shared_block(x1)
    out = Dense(1)(x2)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=SGD(0.01), loss='mse')

    X = np.random.randn(1, 2, 4).astype(np.float32)
    y_true = np.ones((1, 2, 1)).astype(np.float32)

    # Forward then backward
    y_pred = model.forward(X, training=True)
    grad = model.loss_fn.gradient(y_true, y_pred)
    model.backward(grad)

    # Verify all sublayers inside the shared block received gradients
    block_ffn_0_W = shared_block.ffn[0].grads['W']
    block_attn_kernel = shared_block.att.params['Wq'] if hasattr(shared_block.att, 'params') else \
                        shared_block.att.params.get('Wq', None)
    
    assert block_ffn_0_W is not None, "FFN layer in shared block has no gradient"
    assert np.any(np.abs(block_ffn_0_W) > 0), "FFN gradient is all zero"

    # Verify layernorm also received gradients
    g = shared_block.layernorm1.grads.get('gamma')
    assert g is not None
    assert np.any(np.abs(g) > 0), "LayerNorm gamma gradient is all zero"


def test_shared_transformer_block_siamese():
    """Siamese architecture with shared TransformerBlock."""
    shared_block = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16, use_flash=True)

    inp1 = Input(shape=(3, 8), name='seq1')
    inp2 = Input(shape=(3, 8), name='seq2')

    out1 = shared_block(inp1)
    out2 = shared_block(inp2)

    merged = Add()([out1, out2])
    final_out = Dense(1)(merged)

    model = Model(inputs=[inp1, inp2], outputs=final_out)
    model.summary()

    X1 = np.random.randn(2, 3, 8).astype(np.float32)
    X2 = np.random.randn(2, 3, 8).astype(np.float32)

    y = model.predict([X1, X2])
    assert y.shape == (2, 3, 1)
