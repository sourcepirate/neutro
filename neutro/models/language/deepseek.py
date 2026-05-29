import numpy as np
from ..base_model import Sequential
from ...layers.base import Layer
from ...layers.attention.flash_attention import FlashAttention
from ...layers.attention.mla import MultiHeadLatentAttention
from ...layers.normalization.rmsnorm import RMSNorm
from ...layers.core.moe import MoELayer
from ...layers.core.dense import Dense
from ...layers.embedding.embedding import Embedding

class DeepSeekMoEBlock(Layer):
    """
    DeepSeek MoE Block.
    Supports standard Attention or MLA, and MoE.
    """
    def __init__(self, dim, n_heads, n_experts, top_k, use_mla=False, **kwargs):
        super().__init__(**kwargs)
        if use_mla:
            self.attention = MultiHeadLatentAttention(num_heads=n_heads, head_dim=dim//n_heads, latent_dim=dim//2, kv_latent_dim=dim//4)
        else:
            self.attention = FlashAttention(num_heads=n_heads, key_dim=dim, use_rope=True)
            
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        
        # The MoE part
        self.moe = MoELayer(num_experts=n_experts, top_k=top_k, expert_units=dim*2)
        # DeepSeek also has shared experts
        self.shared_expert = Dense(dim)


    def build(self, input_shape):
        self.attention.build(input_shape)
        self.attention_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.moe.build(input_shape)
        self.shared_expert.build(input_shape)
        super().build(input_shape)

    def forward(self, x, training=False, mask=None, kv_cache=None, layer_id=None):
        # Attention
        self.attn_norm_out = self.attention_norm(x, training)
        self.attn_out = self.attention(self.attn_norm_out, mask=mask, training=training, kv_cache=kv_cache, layer_id=layer_id)
        self.h = x + self.attn_out
        
        # MoE + Shared Expert
        self.ffn_norm_out = self.ffn_norm(self.h, training)
        self.moe_out = self.moe(self.ffn_norm_out, training)
        self.shared_out = self.shared_expert(self.ffn_norm_out, training)
        
        return self.h + self.moe_out + self.shared_out

    def backward(self, grad_output):
        # ffn_norm path
        grad_moe = self.moe.backward(grad_output)
        grad_shared = self.shared_expert.backward(grad_output)
        grad_ffn_norm = self.ffn_norm.backward(grad_moe + grad_shared)
        
        # Residual from h
        grad_h = grad_output + grad_ffn_norm
        
        # attention path
        grad_attn = self.attention.backward(grad_h)
        grad_attn_norm = self.attention_norm.backward(grad_attn)
        
        # Residual from x
        grad_x = grad_h + grad_attn_norm
        
        return grad_x

def DeepSeekV1Tiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """
    DeepSeek V1: The strong baseline.
    Standard MHA and MoE.
    """
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
    ])
    for _ in range(n_layers):
        model.add(DeepSeekMoEBlock(dim, n_heads, n_experts=8, top_k=2, use_mla=False))
    model.add(RMSNorm())
    model.add(Dense(vocab_size))
    return model

def DeepSeekV2Tiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """
    DeepSeek V2: The MLA revolution.
    Introduces Multi-head Latent Attention.
    """
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
    ])
    for _ in range(n_layers):
        model.add(DeepSeekMoEBlock(dim, n_heads, n_experts=8, top_k=2, use_mla=True))
    model.add(RMSNorm())
    model.add(Dense(vocab_size))
    return model

def DeepSeekV3Tiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """
    DeepSeek V3: Refined MLA and MoE.
    More experts, more fine-grained routing.
    """
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
    ])
    for _ in range(n_layers):
        model.add(DeepSeekMoEBlock(dim, n_heads, n_experts=16, top_k=4, use_mla=True))
    model.add(RMSNorm())
    model.add(Dense(vocab_size))
    return model

def DeepSeekV4Tiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """
    DeepSeek V4 (Speculative):
    Pushing the limits of MoE and MLA.
    """
    return DeepSeekV3Tiny(vocab_size, seq_len, dim, n_layers + 1, n_heads)

def DeepSeekTiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    # Alias for V2
    return DeepSeekV2Tiny(vocab_size, seq_len, dim, n_layers, n_heads)
