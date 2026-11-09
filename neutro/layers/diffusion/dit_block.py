from .adaln import AdaLNZero
from .time_mlp import TimeMLPEmbedding
from ..base import Layer
from ..attention.flash_attention import FlashAttention
from ..core.swiglu_mlp import SwiGLUMLP


class DiTBlock(Layer):
    """Diffusion Transformer block: AdaLN-Zero + bidirectional attention + SwiGLU.

    x: (B, S, D), cond: (B, C) time/class embedding.
    """

    def __init__(self, dim, n_heads, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.adaln1 = AdaLNZero()
        self.attn = FlashAttention(num_heads=n_heads, key_dim=dim)
        self.adaln2 = AdaLNZero()
        self.mlp = SwiGLUMLP(dim, hidden_dim or dim * 4)

    def build(self, input_shape):
        x_shape, c_shape = input_shape
        self.adaln1.build([x_shape, c_shape])
        self.attn.build(x_shape)
        self.adaln2.build([x_shape, c_shape])
        self.mlp.build(x_shape)
        super().build(x_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def forward(self, inputs, training=False):
        x, cond = inputs
        # Manual AdaLN-Zero path (kept explicit for clarity):
        h_norm = self.adaln1.forward([x, cond], training)
        attn_out = self.attn(h_norm, training=training)
        gate1 = self.adaln1.gate_proj(cond, training)
        while gate1.ndim < attn_out.ndim:
            gate1 = gate1.reshape(gate1.shape[:-1] + (1,) + gate1.shape[-1:])
        h = x + gate1 * attn_out
        h_norm2 = self.adaln2.forward([h, cond], training)
        mlp_out = self.mlp(h_norm2, training)
        gate2 = self.adaln2.gate_proj(cond, training)
        while gate2.ndim < mlp_out.ndim:
            gate2 = gate2.reshape(gate2.shape[:-1] + (1,) + gate2.shape[-1:])
        return h + gate2 * mlp_out
