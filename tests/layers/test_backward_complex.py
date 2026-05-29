import numpy as np
import pytest
from neutro.layers.core.moe import MoELayer
from neutro.layers.normalization.rmsnorm import RMSNorm
from neutro.models.language.llama import LlamaMLP
from neutro.models.language.deepseek import DeepSeekMoEBlock

def test_moe_backward():
    dim = 16
    num_experts = 4
    top_k = 2
    expert_units = 32
    layer = MoELayer(num_experts, top_k, expert_units)
    layer.build((None, 5, dim))
    
    x = np.random.randn(2, 5, dim)
    out = layer(x, training=True)
    grad_output = np.random.randn(*out.shape)
    dx = layer.backward(grad_output)
    
    assert dx.shape == x.shape
    assert 'router_weight' in layer.grads
    assert layer.grads['router_weight'].shape == (dim, num_experts)

def test_rmsnorm_backward():
    dim = 16
    layer = RMSNorm()
    layer.build((None, 5, dim))
    
    x = np.random.randn(2, 5, dim)
    out = layer(x, training=True)
    grad_output = np.random.randn(*out.shape)
    dx = layer.backward(grad_output)
    
    assert dx.shape == x.shape
    assert 'weight' in layer.grads
    assert layer.grads['weight'].shape == (dim,)

def test_llama_mlp_backward():
    dim = 16
    hidden_dim = 64
    layer = LlamaMLP(dim, hidden_dim)
    layer.build((None, 5, dim))
    
    x = np.random.randn(2, 5, dim)
    out = layer(x, training=True)
    grad_output = np.random.randn(*out.shape)
    dx = layer.backward(grad_output)
    
    assert dx.shape == x.shape

def test_deepseek_moe_block_backward():
    dim = 16
    n_heads = 2
    n_experts = 4
    top_k = 2
    layer = DeepSeekMoEBlock(dim, n_heads, n_experts, top_k)
    layer.build((None, 5, dim))
    
    x = np.random.randn(2, 5, dim)
    out = layer(x, training=True)
    grad_output = np.random.randn(*out.shape)
    dx = layer.backward(grad_output)
    
    assert dx.shape == x.shape
