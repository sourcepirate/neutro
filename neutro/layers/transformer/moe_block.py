from ..base import Layer
from ..attention.flash_attention import FlashAttention
from ..normalization.rmsnorm import RMSNorm
from ..core.moe import MoELayer
from ..core.dense import Dense


class MoETransformerBlock(Layer):
    """Generic Attention + MoE block (DeepSeek / Qwen-MoE).

    h = x + Attn(Norm(x)); out = h + MoE(Norm(h)) + SharedExpert(Norm(h)).
    """

    def __init__(self, dim, n_heads, n_experts=8, top_k=2,
                 expert_type='relu', expert_hidden_dim=None,
                 use_shared_expert=True, use_mla=False, **kwargs):
        super().__init__(**kwargs)
        if use_mla:
            from ..attention.mla import MultiHeadLatentAttention
            self.attention = MultiHeadLatentAttention(
                num_heads=n_heads, head_dim=dim // n_heads,
                latent_dim=dim // 2, kv_latent_dim=dim // 4)
        else:
            self.attention = FlashAttention(num_heads=n_heads, key_dim=dim,
                                            use_rope=True)
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        self.moe = MoELayer(num_experts=n_experts, top_k=top_k,
                            expert_units=expert_hidden_dim or dim * 2,
                            expert_type=expert_type,
                            expert_hidden_dim=expert_hidden_dim or dim * 2)
        self.use_shared = use_shared_expert
        self.shared_expert = Dense(dim) if use_shared_expert else None

    def build(self, input_shape):
        self.attention.build(input_shape)
        self.attention_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.moe.build(input_shape)
        if self.shared_expert is not None:
            self.shared_expert.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, x, training=False, mask=None, kv_cache=None, layer_id=None):
        try:
            attn_out = self.attention(
                self.attention_norm(x, training), mask=mask,
                training=training, kv_cache=kv_cache, layer_id=layer_id)
        except TypeError:
            attn_out = self.attention(
                self.attention_norm(x, training), training=training)
        h = x + attn_out
        ffn_norm = self.ffn_norm(h, training)
        out = h + self.moe(ffn_norm, training)
        if self.shared_expert is not None:
            out = out + self.shared_expert(ffn_norm, training)
        return out

    def aux_loss(self):
        return self.moe.aux_loss()
