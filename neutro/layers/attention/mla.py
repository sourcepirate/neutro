import numpy as np
from ..base import Layer
from ..core.dense import Dense
from neutro.autograd import Tensor, ops as autograd_ops


class MultiHeadLatentAttention(Layer):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2.

    Compresses the input into a low-rank latent vector for both Q and KV,
    then decompresses to full multi-head representations.  KV caching happens
    on the *latent* vector (saving memory), not on the full K/V.

    Q path:   x -> Dense(latent_dim) -> Dense(H*d)
    KV path:  x -> Dense(kv_latent_dim) -> [cache] -> Dense(H*2d) -> split K,V

    Shapes:
        x: (B, S, E)
        q: (B, H, S, d)          after compress + decompress + reshape + transpose
        kv: (B, total_S, H, 2d)   after KV decompress + reshape
        k, v: (B, H, total_S, d)  after split + transpose
        scores: (B, H, S, total_S) = q @ k^T / sqrt(d)
        out: (B, S, E)            after attn @ v + transpose + project
    """
    def __init__(self, num_heads, head_dim, latent_dim, kv_latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.kv_latent_dim = kv_latent_dim
        self.scale = float(1.0 / np.sqrt(head_dim))

    def build(self, input_shape):
        self.embed_dim = input_shape[-1]

        self.kv_compress = Dense(self.kv_latent_dim, use_bias=False)
        self.kv_compress.build(input_shape)

        self.kv_decompress = Dense(self.num_heads * (self.head_dim + self.head_dim))
        self.kv_decompress.build((None, self.kv_latent_dim))

        self.q_compress = Dense(self.latent_dim, use_bias=False)
        self.q_compress.build(input_shape)

        self.q_decompress = Dense(self.num_heads * self.head_dim)
        self.q_decompress.build((None, self.latent_dim))

        self.wo = Dense(self.embed_dim, use_bias=False)
        self.wo.build((None, self.num_heads * self.head_dim))

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, x, mask=None, training=False, kv_cache=None, layer_id=None):
        B, S, E = x.shape
        H = self.num_heads
        d = self.head_dim

        q_latent = self.q_compress(x, training=training)
        q = self.q_decompress(q_latent, training=training)
        q = q.reshape(B, S, H, d).transpose(0, 2, 1, 3)

        kv_latent = self.kv_compress(x, training=training)

        if kv_cache is not None and layer_id is not None:
            kv_latent_4d = kv_latent[:, None, :, :]
            _, kv_latent_cached = kv_cache.update(kv_latent_4d, kv_latent_4d, layer_id)
            kv_latent = kv_latent_cached[:, 0, :, :]

        kv = self.kv_decompress(kv_latent, training=training)
        kv = kv.reshape(B, -1, H, 2 * d)
        k = kv[..., :d].transpose(0, 2, 1, 3)
        v = kv[..., d:].transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + Tensor(-1e9) * mask

        attn_weights = autograd_ops.softmax(scores, axis=-1)
        attn_out = attn_weights @ v

        out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, H * d)
        return self.wo(out, training=training)
