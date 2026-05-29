import numpy as np
import pytest
from neutro.layers import MultiHeadAttention, FlashAttention

def test_flash_attention_parity():
    batch, seq_len, embed_dim = 2, 16, 32
    num_heads = 4
    key_dim = 32
    
    x = np.random.randn(batch, seq_len, embed_dim)
    
    # Initialize both layers
    mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    flash = FlashAttention(num_heads=num_heads, key_dim=key_dim, block_size_r=4, block_size_c=4)
    
    mha.build(x.shape)
    flash.build(x.shape)
    
    # Force identical weights for comparison
    flash.params['Wq'] = mha.params['Wq'].copy()
    flash.params['Wk'] = mha.params['Wk'].copy()
    flash.params['Wv'] = mha.params['Wv'].copy()
    flash.params['Wo'] = mha.params['Wo'].copy()
    
    out_mha = mha.forward(x)
    out_flash = flash.forward(x)
    
    # Check parity
    np.testing.assert_allclose(out_mha, out_flash, atol=1e-5)

def test_flash_attention_gradient_parity():
    batch, seq_len, embed_dim = 2, 8, 16
    num_heads = 2
    key_dim = 16
    
    x = np.random.randn(batch, seq_len, embed_dim)
    grad_out = np.random.randn(batch, seq_len, embed_dim)
    
    # Initialize both layers
    mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    flash = FlashAttention(num_heads=num_heads, key_dim=key_dim, block_size_r=4, block_size_c=4)
    
    mha.build(x.shape)
    flash.build(x.shape)
    
    # Force identical weights
    flash.params['Wq'] = mha.params['Wq'].copy()
    flash.params['Wk'] = mha.params['Wk'].copy()
    flash.params['Wv'] = mha.params['Wv'].copy()
    flash.params['Wo'] = mha.params['Wo'].copy()
    
    # Forward
    out_mha = mha.forward(x)
    out_flash = flash.forward(x)
    
    # Backward
    dx_mha = mha.backward(grad_out)
    dx_flash = flash.backward(grad_out)
    
    # Check input gradients parity
    np.testing.assert_allclose(dx_mha, dx_flash, atol=1e-5)
    
    # Check weight gradients parity
    for p in ['Wq', 'Wk', 'Wv', 'Wo']:
        np.testing.assert_allclose(mha.grads[p], flash.grads[p], atol=1e-5, err_msg=f"Gradient mismatch for {p}")
