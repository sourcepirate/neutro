import numpy as np
import pytest
from neutro.layers.attention.kv_cache import KVCache
from neutro.layers.attention.flash_attention import FlashAttention
from neutro.layers.attention.mla import MultiHeadLatentAttention
from neutro.models.language.llama import LlamaTiny
from neutro.models.language.deepseek import DeepSeekV2Tiny
from neutro.models.language.gpt import GPT2

def test_kv_cache_basic():
    cache = KVCache()
    k = np.random.rand(1, 4, 1, 16)
    v = np.random.rand(1, 4, 1, 16)
    
    k_out, v_out = cache.update(k, v, layer_id=0)
    assert k_out.shape == (1, 4, 1, 16)
    assert v_out.shape == (1, 4, 1, 16)
    
    k2 = np.random.rand(1, 4, 1, 16)
    v2 = np.random.rand(1, 4, 1, 16)
    k_out, v_out = cache.update(k2, v2, layer_id=0)
    assert k_out.shape == (1, 4, 2, 16)
    assert v_out.shape == (1, 4, 2, 16)
    
    cache.reset()
    assert len(cache.k_cache) == 0

def test_flash_attention_with_cache():
    layer = FlashAttention(num_heads=2, key_dim=16, use_rope=True)
    layer.build((1, 5, 16))
    
    cache = KVCache()
    x = np.random.rand(1, 1, 16)
    
    # Step 1
    out1 = layer(x, kv_cache=cache, layer_id=0)
    assert out1.shape == (1, 1, 16)
    assert cache.k_cache[0].shape == (1, 2, 1, 8)
    
    # Step 2
    out2 = layer(x, kv_cache=cache, layer_id=0)
    assert out2.shape == (1, 1, 16)
    assert cache.k_cache[0].shape == (1, 2, 2, 8)

def test_mla_with_cache():
    layer = MultiHeadLatentAttention(num_heads=2, head_dim=8, latent_dim=16, kv_latent_dim=8)
    layer.build((1, 5, 16))
    
    cache = KVCache()
    x = np.random.rand(1, 1, 16)
    
    # Step 1
    out1 = layer(x, kv_cache=cache, layer_id=0)
    assert out1.shape == (1, 1, 16)
    # MLA caches latent vector (B, 1, S, D)
    assert cache.k_cache[0].shape == (1, 1, 1, 8)
    
    # Step 2
    out2 = layer(x, kv_cache=cache, layer_id=0)
    assert out2.shape == (1, 1, 16)
    assert cache.k_cache[0].shape == (1, 1, 2, 8)

def test_model_generate():
    vocab_size = 50
    seq_len = 10
    model = LlamaTiny(vocab_size, seq_len, dim=32, n_layers=1, n_heads=2)
    
    start_tokens = np.random.randint(0, vocab_size, (1, 3))
    generated = model.generate(start_tokens, max_new_tokens=4)
    
    assert generated.shape == (1, 7)
    assert np.all(generated[:, :3] == start_tokens)

def test_gpt2_generate():
    vocab_size = 50
    seq_len = 10
    model = GPT2(vocab_size, seq_len, dim=32, n_layers=1, n_heads=2)
    
    start_tokens = np.random.randint(0, vocab_size, (1, 3))
    generated = model.generate(start_tokens, max_new_tokens=4)
    
    assert generated.shape == (1, 7)

def test_deepseek_v2_generate():
    vocab_size = 50
    seq_len = 10
    model = DeepSeekV2Tiny(vocab_size, seq_len, dim=32, n_layers=1, n_heads=2)
    
    start_tokens = np.random.randint(0, vocab_size, (1, 3))
    generated = model.generate(start_tokens, max_new_tokens=4)
    
    assert generated.shape == (1, 7)
