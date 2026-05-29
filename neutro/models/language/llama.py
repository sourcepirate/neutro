import numpy as np
from ..base_model import Sequential
from ...layers.base import Layer
from ...layers.attention.flash_attention import FlashAttention
from ...layers.normalization.rmsnorm import RMSNorm
from ...layers.core.dense import Dense
from ...layers.core.activation import Activation
from ...layers.embedding.embedding import Embedding
from ...activations.silu import SiLU
from ...utils.rope_utils import precompute_freqs_cis, apply_rotary_emb

class LlamaMLP(Layer):
    """
    The Llama MLP using SwiGLU. 
    It's basically a three-way Dense layer party.
    """
    def __init__(self, dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.w1 = Dense(hidden_dim)
        self.w2 = Dense(dim)
        self.w3 = Dense(hidden_dim)
        self.silu = SiLU()

    def build(self, input_shape):
        self.w1.build(input_shape)
        self.w2.build((input_shape[0], input_shape[1], self.w1.units))
        self.w3.build(input_shape)
        super().build(input_shape)

    def forward(self, x, training=False):
        self.x = x
        # SwiGLU: (SiLU(x @ w1) * (x @ w3)) @ w2
        self.gate = self.w1(x, training)
        self.activated_gate = self.silu(self.gate)
        self.value = self.w3(x, training)
        self.multiplied = self.activated_gate * self.value
        return self.w2(self.multiplied, training)

    def backward(self, grad_output):
        # 1. Backprop through w2
        # grad_output is (batch, seq, dim)
        grad_multiplied = self.w2.backward(grad_output)
        
        # 2. Backprop through element-wise multiplication
        # multiplied = activated_gate * value
        grad_activated_gate = grad_multiplied * self.value
        grad_value = grad_multiplied * self.activated_gate
        
        # 3. Backprop through SiLU
        grad_gate = grad_activated_gate * self.silu.gradient(self.gate)
        
        # 4. Backprop through w1 and w3
        grad_x_w1 = self.w1.backward(grad_gate)
        grad_x_w3 = self.w3.backward(grad_value)
        
        # 5. Total grad_x
        return grad_x_w1 + grad_x_w3

class LlamaBlock(Layer):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.dim = dim
        self.attention = FlashAttention(num_heads=n_heads, key_dim=dim, use_rope=True)
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        self.feed_forward = LlamaMLP(dim, hidden_dim)

    def build(self, input_shape):
        self.attention.build(input_shape)
        self.attention_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.feed_forward.build(input_shape)
        super().build(input_shape)

    def forward(self, x, training=False, mask=None, kv_cache=None, layer_id=None):
        # Residual 1
        self.h1_norm = self.attention_norm(x, training)
        self.attn_out = self.attention(self.h1_norm, mask=mask, training=training, kv_cache=kv_cache, layer_id=layer_id)
        self.h = x + self.attn_out
        
        # Residual 2
        self.h2_norm = self.ffn_norm(self.h, training)
        self.ffn_out = self.feed_forward(self.h2_norm, training)
        out = self.h + self.ffn_out
        return out

    def backward(self, grad_output):
        # Residual 2 backward
        # out = h + ffn(norm(h))
        grad_ffn_out = self.feed_forward.backward(grad_output)
        grad_h2_norm = self.ffn_norm.backward(grad_ffn_out)
        grad_h = grad_output + grad_h2_norm
        
        # Residual 1 backward
        # h = x + attn(norm(x))
        grad_attn_out = self.attention.backward(grad_h)
        grad_h1_norm = self.attention_norm.backward(grad_attn_out)
        grad_x = grad_h + grad_h1_norm
        
        return grad_x

def LlamaTiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """
    Llama: The open-source king.
    Uses RMSNorm, RoPE (simulated here in Block), and SwiGLU.
    """
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
    ])
    
    for _ in range(n_layers):
        model.add(LlamaBlock(dim, n_heads, n_heads, dim // n_heads, dim * 4))
        
    model.add(RMSNorm())
    model.add(Dense(vocab_size))
    return model
