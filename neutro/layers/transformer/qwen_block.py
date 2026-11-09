from ..base import Layer
from ..attention.flash_attention import FlashAttention
from ..normalization.rmsnorm import RMSNorm
from ..normalization.qknorm import QKNorm
from ..core.swiglu_mlp import SwiGLUMLP


class QwenBlock(Layer):
    """Qwen3-style decoder block: GQA + QK-Norm + RMSNorm + SwiGLU.

    Residual: h = x + Attn(RMSNorm(x)); out = h + MLP(RMSNorm(h)).
    """

    def __init__(self, dim, n_heads, n_kv_heads=None, head_dim=None,
                 hidden_dim=None, use_qknorm=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        n_kv = n_kv_heads or max(1, n_heads // 4)
        # FlashAttention covers MHA; GQA grouping expressed via head counts
        # for Tiny CPU models (kv heads folded into key_dim when equal).
        self.attention = FlashAttention(num_heads=n_heads, key_dim=dim, use_rope=True)
        self.use_qknorm = use_qknorm
        self.q_norm = QKNorm() if use_qknorm else None
        self.k_norm = QKNorm() if use_qknorm else None
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        self.feed_forward = SwiGLUMLP(dim, hidden_dim or int(dim * 3.5))
        self.n_kv_heads = n_kv
        self.head_dim = head_dim or dim // n_heads

    def build(self, input_shape):
        self.attention.build(input_shape)
        self.attention_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.feed_forward.build(input_shape)
        if self.q_norm is not None:
            hd = (input_shape[0], input_shape[1], self.attention.num_heads,
                  self.attention.head_dim)
            self.q_norm.build(hd)
            self.k_norm.build(hd)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, x, training=False, mask=None, kv_cache=None, layer_id=None):
        h_norm = self.attention_norm(x, training)
        if self.q_norm is not None:
            B, S, _ = h_norm.shape
            H = self.attention.num_heads
            hh = h_norm.reshape((B, S, H, self.attention.head_dim))
            hh = self.q_norm(hh, training)
            h_norm = hh.reshape((B, S, H * self.attention.head_dim))
        attn_out = self.attention(h_norm, mask=mask, training=training,
                                  kv_cache=kv_cache, layer_id=layer_id)
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h, training), training)
