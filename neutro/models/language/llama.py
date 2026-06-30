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

    SwiGLU(x) = (SiLU(x @ W1) * (x @ W3)) @ W2

    Shapes:
        x: (B, S, D) → gate: (B, S, U) → SiLU → (B, S, U)
        value: (B, S, U) → mul: (B, S, U) → (B, S, D)
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
        gate = self.w1(x, training)
        activated_gate = self.silu(gate)
        value = self.w3(x, training)
        multiplied = activated_gate * value
        return self.w2(multiplied, training)

class LlamaBlock(Layer):
    """
    A single Llama decoder block with Pre-Norm, RoPE attention, and SwiGLU FFN.

    Residual path:
        h  = x + Attention(Norm(x))
        out = h + SwiGLU(Norm(h))
    """
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
        h1_norm = self.attention_norm(x, training)
        attn_out = self.attention(h1_norm, mask=mask, training=training, kv_cache=kv_cache, layer_id=layer_id)
        h = x + attn_out

        h2_norm = self.ffn_norm(h, training)
        ffn_out = self.feed_forward(h2_norm, training)
        out = h + ffn_out
        return out

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
