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
    DeepSeek MoE Block with optional MLA and shared expert.

    Residual path:
        h   = x + Attention(Norm(x))
        out = h + MoE(Norm(h)) + SharedExpert(Norm(h))
    """
    def __init__(self, dim, n_heads, n_experts, top_k, use_mla=False, **kwargs):
        super().__init__(**kwargs)
        if use_mla:
            self.attention = MultiHeadLatentAttention(num_heads=n_heads, head_dim=dim//n_heads, latent_dim=dim//2, kv_latent_dim=dim//4)
        else:
            self.attention = FlashAttention(num_heads=n_heads, key_dim=dim, use_rope=True)

        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()

        self.moe = MoELayer(num_experts=n_experts, top_k=top_k, expert_units=dim*2,
                            expert_type='swiglu', expert_hidden_dim=dim*2)
        self.shared_expert = Dense(dim)

    def build(self, input_shape):
        self.attention.build(input_shape)
        self.attention_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.moe.build(input_shape)
        self.shared_expert.build(input_shape)
        super().build(input_shape)

    def forward(self, x, training=False, mask=None, kv_cache=None, layer_id=None):
        attn_norm_out = self.attention_norm(x, training)
        attn_out = self.attention(attn_norm_out, mask=mask, training=training, kv_cache=kv_cache, layer_id=layer_id)
        h = x + attn_out

        ffn_norm_out = self.ffn_norm(h, training)
        moe_out = self.moe(ffn_norm_out, training)
        shared_out = self.shared_expert(ffn_norm_out, training)

        return h + moe_out + shared_out

def _deepseek_tiny(vocab_size, seq_len, dim, n_layers, n_heads, n_experts, top_k, use_mla):
    model = Sequential([Embedding(vocab_size, dim, input_shape=(seq_len,))])
    for _ in range(n_layers):
        model.add(DeepSeekMoEBlock(
            dim, n_heads, n_experts=n_experts, top_k=top_k, use_mla=use_mla
        ))
    model.add(RMSNorm())
    model.add(Dense(vocab_size))
    return model


def DeepSeekV1Tiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """DeepSeek V1: standard MHA and MoE."""
    return _deepseek_tiny(vocab_size, seq_len, dim, n_layers, n_heads, 8, 2, False)


def DeepSeekV2Tiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """DeepSeek V2: MLA plus MoE."""
    return _deepseek_tiny(vocab_size, seq_len, dim, n_layers, n_heads, 8, 2, True)


def DeepSeekV3Tiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """DeepSeek V3: more experts and a larger top-k."""
    return _deepseek_tiny(vocab_size, seq_len, dim, n_layers, n_heads, 16, 4, True)


def DeepSeekV4Tiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """Alias for a deeper DeepSeekV3Tiny."""
    return DeepSeekV3Tiny(vocab_size, seq_len, dim, n_layers + 1, n_heads)


def DeepSeekTiny(vocab_size, seq_len, dim=512, n_layers=2, n_heads=8):
    """Alias for DeepSeekV2Tiny."""
    return DeepSeekV2Tiny(vocab_size, seq_len, dim, n_layers, n_heads)
