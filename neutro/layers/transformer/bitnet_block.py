import numpy as np
from ..base import Layer
from ..core.bitlinear import BitLinear
from ..normalization.rmsnorm import RMSNorm
from neutro.autograd import ops as autograd_ops


class BitNetBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, mode='b1.58', activation_bits=8, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.mode = mode
        self.activation_bits = activation_bits
        self.dropout_rate = dropout
        self.scale = float(1.0 / np.sqrt(self.head_dim))

        self.wq = BitLinear(embed_dim, mode=mode, activation_bits=activation_bits)
        self.wk = BitLinear(embed_dim, mode=mode, activation_bits=activation_bits)
        self.wv = BitLinear(embed_dim, mode=mode, activation_bits=activation_bits)
        self.wo = BitLinear(embed_dim, mode=mode, activation_bits=activation_bits)

        self.ffn_gate = BitLinear(ff_dim, mode=mode, activation_bits=activation_bits)
        self.ffn_up = BitLinear(ff_dim, mode=mode, activation_bits=activation_bits)
        self.ffn_down = BitLinear(embed_dim, mode=mode, activation_bits=activation_bits)

        self.attn_norm = RMSNorm()
        self.ffn_norm = RMSNorm()

    def build(self, input_shape):
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        self.wo.build(input_shape)
        self.ffn_gate.build(input_shape)
        self.ffn_up.build(input_shape)
        gate_shape = (input_shape[0], input_shape[1], self.ff_dim)
        self.ffn_down.build(gate_shape)
        self.attn_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _split_heads(self, x):
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x, batch, seq):
        return x.transpose(0, 2, 1, 3).reshape(batch, seq, self.embed_dim)

    @staticmethod
    def _swiglu(gate, up):
        return gate * autograd_ops.sigmoid(gate) * up

    def _attention(self, inputs, training, mask):
        batch, seq, _ = inputs.shape
        normed = self.attn_norm(inputs, training=training)
        query = self._split_heads(self.wq(normed, training=training))
        key = self._split_heads(self.wk(normed, training=training))
        value = self._split_heads(self.wv(normed, training=training))

        scores = self.scale * (query @ key.transpose(0, 1, 3, 2))
        if mask is not None:
            scores = scores + mask * -1e9

        weights = autograd_ops.softmax(scores, axis=-1)
        output = self._merge_heads(weights @ value, batch, seq)
        return inputs + self.wo(output, training=training)

    def _feed_forward(self, inputs, training):
        normed = self.ffn_norm(inputs, training=training)
        gate = self.ffn_gate(normed, training=training)
        up = self.ffn_up(normed, training=training)
        return inputs + self.ffn_down(self._swiglu(gate, up), training=training)

    def forward(self, inputs, training=False, mask=None, kv_cache=None, layer_id=None):
        return self._feed_forward(self._attention(inputs, training, mask), training)
