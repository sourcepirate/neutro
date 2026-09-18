import numpy as np
import pytest
from neutro.layers.core.qlora import QLoRADense, apply_qlora
from neutro.layers.core.dense import Dense
from neutro.models import Sequential
from neutro.utils.quantization import (
    quantize_nf4, dequantize_nf4,
    quantize_absmax, dequantize_absmax,
    NF4_CODEBOOK,
)


def test_nf4_roundtrip():
    np.random.seed(0)
    W = np.random.randn(64, 32).astype(float)
    codes, absmax, shape = quantize_nf4(W, block_size=64)
    W_dq = dequantize_nf4(codes, absmax, block_size=64, shape=shape)
    assert W_dq.shape == W.shape
    # NF4 error should be modest
    assert np.mean(np.abs(W - W_dq)) < 0.15
    assert codes.dtype == np.uint8
    assert np.all(codes < 16)


def test_nf4_codebook_stats():
    assert NF4_CODEBOOK.shape == (16,)
    assert NF4_CODEBOOK[0] == -1.0
    assert NF4_CODEBOOK[-1] == 1.0
    assert NF4_CODEBOOK[7] == 0.0


def test_double_quant_roundtrip():
    np.random.seed(1)
    W = np.random.randn(128, 32).astype(float)
    _, absmax, _ = quantize_nf4(W, block_size=64)
    q, dq, scales2, shp = quantize_absmax(absmax, block_size=256)
    absmax_dq = dequantize_absmax(q, scales2, block_size=256, shape=shp)
    assert absmax_dq.shape == absmax.shape
    assert np.mean(np.abs(absmax - absmax_dq)) < 0.02


def test_qlora_forward_shape():
    layer = QLoRADense(10, rank=4)
    x = np.random.randn(5, 8)
    out = layer(x)
    assert out.shape == (5, 10)


def test_qlora_3d_forward():
    layer = QLoRADense(6, rank=2)
    x = np.random.randn(2, 3, 8)
    out = layer(x)
    assert out.shape == (2, 3, 6)


def test_qlora_backward_shapes():
    layer = QLoRADense(10, rank=4, use_bias=True)
    x = np.random.randn(5, 8)
    out = layer(x)
    grad = np.random.randn(5, 10)
    dx = layer.backward(grad)
    assert dx.shape == (5, 8)
    assert 'lora_A' in layer.grads
    assert 'lora_B' in layer.grads
    assert layer.grads['lora_A'].shape == (8, 4)
    assert layer.grads['lora_B'].shape == (4, 10)


def test_qlora_from_dense():
    dense = Dense(10, use_bias=True)
    x = np.random.randn(3, 8)
    dense(x)
    qlora = QLoRADense.from_dense(dense, rank=4, block_size=64, double_quant=True)
    # Dequantized weight should be close to original
    W_dq = qlora._dequant_weight()
    assert W_dq.shape == (8, 10)
    assert np.mean(np.abs(W_dq - dense.params['W'].data)) < 0.15


def test_qlora_count_params():
    layer = QLoRADense(10, rank=4)
    x = np.random.randn(2, 8)
    layer(x)
    # Only LoRA params counted
    assert layer.count_params() == 8 * 4 + 4 * 10


def test_qlora_memory_bytes():
    layer = QLoRADense(32, rank=4, block_size=64, double_quant=True)
    x = np.random.randn(2, 16)
    layer(x)
    stats = layer.memory_bytes()
    # NF4 base should be ~0.5 bytes/param
    n = layer.W_codes.size  # 16*32=512
    assert stats["base_nf4"] == n * 0.5
    assert stats["total"] > 0


def test_qlora_double_quant_flag():
    layer_dq = QLoRADense(10, rank=4, block_size=64, double_quant=True)
    layer_no = QLoRADense(10, rank=4, block_size=64, double_quant=False)
    x = np.random.randn(2, 8)
    layer_dq(x)
    layer_no(x)
    assert layer_dq.double_quant is True
    assert layer_no.double_quant is False
    # Both should forward without error
    out_dq = layer_dq(np.random.randn(2, 8))
    out_no = layer_no(np.random.randn(2, 8))
    assert out_dq.shape == (2, 10)
    assert out_no.shape == (2, 10)


def test_qlora_activation():
    layer = QLoRADense(10, rank=2, activation='relu', use_bias=False)
    x = np.random.randn(4, 6)
    out = layer(x)
    out_np = out.data if hasattr(out, 'data') else out
    assert np.all(out_np >= -1e-7)
    dx = layer.backward(np.random.randn(4, 10))
    assert dx.shape == (4, 6)


def test_qlora_scaling():
    layer = QLoRADense(8, rank=4, alpha=16)
    assert layer.scaling == 4.0


def test_apply_qlora_sequential():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(8,)),
        Dense(4),
    ])
    x = np.random.randn(2, 8)
    model(x)
    model, adapted = apply_qlora(model, rank=2, block_size=64, double_quant=False, verbose=False)
    assert len(adapted) == 2
    assert all(isinstance(l, QLoRADense) for l in adapted)
    out = model(x)
    assert out.shape == (2, 4)


def test_apply_qlora_target_modules():
    model = Sequential([
        Dense(16, input_shape=(8,), name="hidden"),
        Dense(4, name="output"),
    ])
    model(np.random.randn(2, 8))
    model, adapted = apply_qlora(model, rank=2, target_modules=["output"], verbose=False)
    assert len(adapted) == 1
    assert adapted[0].name == "output_qlora"


def test_qlora_no_nan():
    layer = QLoRADense(10, rank=4, block_size=64, double_quant=True)
    x = np.random.randn(5, 8) * 10  # large input
    out = layer(x)
    out_np = out.data if hasattr(out, 'data') else out
    assert not np.any(np.isnan(out_np))
    assert not np.any(np.isinf(out_np))


def test_qlora_backward_after_nonzero_B():
    np.random.seed(3)
    layer = QLoRADense(6, rank=2, alpha=4, use_bias=False, block_size=64)
    x = np.random.randn(3, 5)
    layer(x)
    layer.params['lora_B'].data[:] = np.random.randn(2, 6) * 0.5
    out = layer(x)
    yd = out.data if hasattr(out, 'data') else out
    grad_out = np.random.randn(*yd.shape)
    layer.backward(grad_out)
    assert np.any(np.abs(layer.grads['lora_A']) > 1e-8)


def test_qlora_get_merged_weight():
    layer = QLoRADense(8, rank=2, alpha=4, block_size=64)
    x = np.random.randn(2, 6)
    layer(x)
    W_merged = layer.get_merged_weight()
    assert W_merged.shape == (6, 8)
    # With B=0, merged == dequant
    W_dq = layer._dequant_weight()
    assert np.allclose(W_merged, W_dq, atol=1e-7)
