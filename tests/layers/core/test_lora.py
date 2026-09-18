import numpy as np
import pytest
from neutro.layers.core.lora import LoRADense, apply_lora
from neutro.layers.core.dense import Dense
from neutro.models import Sequential
from neutro.utils.quantization import quantize_nf4, dequantize_nf4


def test_lora_forward_shape():
    layer = LoRADense(10, rank=4)
    x = np.random.randn(5, 8)
    out = layer(x)
    assert out.shape == (5, 10)


def test_lora_3d_forward():
    layer = LoRADense(6, rank=2)
    x = np.random.randn(2, 3, 8)
    out = layer(x)
    assert out.shape == (2, 3, 6)


def test_lora_initial_delta_zero():
    np.random.seed(0)
    layer = LoRADense(10, rank=4, alpha=8)
    x = np.random.randn(4, 8)
    # Build
    layer(x)
    # With B=0, lora path should be zero => output == x @ W_frozen (+ bias if any)
    # Compute expected base
    base = x @ layer.W_frozen
    if layer.b_frozen is not None:
        base = base + layer.b_frozen
    out = layer(x)
    out_np = out.data if hasattr(out, 'data') else out
    assert np.allclose(out_np, base, atol=1e-7)


def test_lora_backward_shapes():
    layer = LoRADense(10, rank=4, use_bias=True)
    x = np.random.randn(5, 8)
    out = layer(x)
    grad = np.random.randn(5, 10)
    dx = layer.backward(grad)
    assert dx.shape == (5, 8)
    assert 'lora_A' in layer.grads
    assert 'lora_B' in layer.grads
    assert layer.grads['lora_A'].shape == (8, 4)
    assert layer.grads['lora_B'].shape == (4, 10)
    # Frozen weights must NOT be in grads
    assert 'W' not in layer.grads
    assert 'b' not in layer.grads


def test_lora_backward_after_training():
    """After B becomes non-zero, grad_A should be non-zero too."""
    np.random.seed(1)
    layer = LoRADense(6, rank=2, alpha=4, use_bias=False)
    x = np.random.randn(3, 5)
    layer(x)
    # Make B non-zero
    layer.params['lora_B'].data[:] = np.random.randn(2, 6) * 0.5
    out = layer(x)
    yd = out.data if hasattr(out, 'data') else out
    grad_out = 2 * yd
    layer.backward(grad_out)
    assert np.any(np.abs(layer.grads['lora_A']) > 1e-8)
    assert np.any(np.abs(layer.grads['lora_B']) > 1e-8)


def test_lora_count_params():
    layer = LoRADense(10, rank=4)
    x = np.random.randn(2, 8)
    layer(x)
    # Trainable params: A (8*4) + B (4*10) = 72
    assert layer.count_params() == 8 * 4 + 4 * 10
    # Frozen
    assert layer.count_params_frozen() == 8 * 10 + 10 + 72  # W + b + LoRA


def test_lora_from_dense_copies_weights():
    dense = Dense(10, use_bias=True)
    x = np.random.randn(3, 8)
    dense(x)
    lora = LoRADense.from_dense(dense, rank=4, alpha=8)
    assert np.allclose(lora.W_frozen, dense.params['W'].data)
    assert np.allclose(lora.b_frozen, dense.params['b'].data)
    # Forward with B=0 should match dense (no activation)
    out_dense = dense(x)
    out_lora = lora(x)
    d_np = out_dense.data if hasattr(out_dense, 'data') else out_dense
    l_np = out_lora.data if hasattr(out_lora, 'data') else out_lora
    assert np.allclose(d_np, l_np, atol=1e-7)


def test_lora_merge_weights():
    np.random.seed(2)
    layer = LoRADense(6, rank=2, alpha=4, use_bias=False)
    x = np.random.randn(2, 5)
    layer(x)
    layer.params['lora_B'].data[:] = np.random.randn(2, 6) * 0.3
    out_before = layer(x)
    before_np = out_before.data if hasattr(out_before, 'data') else out_before
    layer.merge_weights()
    out_after = layer(x)
    after_np = out_after.data if hasattr(out_after, 'data') else out_after
    assert np.allclose(before_np, after_np, atol=1e-7)
    # After merge, B should be zero
    assert np.allclose(layer.params['lora_B'].data, 0)


def test_lora_scaling():
    layer = LoRADense(8, rank=4, alpha=8)
    assert layer.scaling == 2.0
    layer2 = LoRADense(8, rank=2, alpha=16)
    assert layer2.scaling == 8.0


def test_lora_activation():
    layer = LoRADense(10, rank=2, activation='relu', use_bias=False)
    x = np.random.randn(4, 6)
    out = layer(x)
    # ReLU: all outputs >=0 (since activation applied after sum)
    out_np = out.data if hasattr(out, 'data') else out
    assert np.all(out_np >= -1e-7)
    # Backward should still work
    grad = np.random.randn(4, 10)
    dx = layer.backward(grad)
    assert dx.shape == (4, 6)


def test_apply_lora_sequential():
    np.random.seed(0)
    model = Sequential([
        Dense(16, activation='relu', input_shape=(8,)),
        Dense(4),
    ])
    x = np.random.randn(2, 8)
    model(x)
    out_before = np.asarray(model.predict(x))
    model, adapted = apply_lora(model, rank=2, alpha=4, verbose=False)
    assert len(adapted) == 2
    assert all(isinstance(l, LoRADense) for l in adapted)
    # Forward with B=0 should be close to before (weights copied)
    out_after = np.asarray(model.predict(x))
    assert np.allclose(out_before, out_after, atol=1e-7)


def test_apply_lora_target_modules():
    model = Sequential([
        Dense(16, input_shape=(8,), name="hidden"),
        Dense(4, name="output"),
    ])
    model(np.random.randn(2, 8))
    # Only adapt output
    model, adapted = apply_lora(model, rank=2, target_modules=["output"], verbose=False)
    assert len(adapted) == 1
    assert adapted[0].name == "output_lora"


def test_apply_lora_with_transformer():
    from neutro.layers import Embedding, TransformerBlock, Softmax
    model = Sequential([
        Embedding(50, 16, input_shape=(8,)),
        TransformerBlock(embed_dim=16, num_heads=2, ff_dim=32, causal=False),
        Dense(50),
        Softmax(),
    ])
    x = np.random.randint(0, 50, size=(2, 8))
    model(x)
    model, adapted = apply_lora(model, rank=4, verbose=False)
    # At least the FFN + output Dense should be adapted
    assert len(adapted) >= 2


def test_lora_invalid_rank():
    with pytest.raises(ValueError):
        LoRADense(10, rank=0)


def test_lora_dropout():
    layer = LoRADense(10, rank=4, dropout=0.5)
    x = np.random.randn(5, 8)
    layer(x)  # build
    layer.params['lora_B'].data[:] = np.ones((4, 10))
    out_train = layer(x, training=True)
    out_eval = layer(x, training=False)
    # With dropout, train vs eval should differ when B non-zero
    t_np = out_train.data if hasattr(out_train, 'data') else out_train
    e_np = out_eval.data if hasattr(out_eval, 'data') else out_eval
    assert not np.allclose(t_np, e_np)
