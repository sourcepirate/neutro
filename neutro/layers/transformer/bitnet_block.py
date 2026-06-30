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

    def forward(self, inputs, training=False, mask=None, kv_cache=None, layer_id=None):
        B, S, D = inputs.shape
        H = self.num_heads
        Dh = self.head_dim

        h1_norm = self.attn_norm(inputs, training=training)

        Q = self.wq(h1_norm, training=training)
        K = self.wk(h1_norm, training=training)
        V = self.wv(h1_norm, training=training)

        Q = Q.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        K = K.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

        scores = self.scale * (Q @ K.transpose(0, 1, 3, 2))

        if mask is not None:
            scores = scores + mask * -1e9

        attn_weights = autograd_ops.softmax(scores, axis=-1)

        attn_out = attn_weights @ V
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)

        attn_proj = self.wo(attn_out, training=training)

        h = inputs + attn_proj

        h2_norm = self.ffn_norm(h, training=training)

        gate = self.ffn_gate(h2_norm, training=training)
        up = self.ffn_up(h2_norm, training=training)

        sigmoid_gate = autograd_ops.sigmoid(gate)
        activated_gate = gate * sigmoid_gate

        multiplied = activated_gate * up

        ffn_out = self.ffn_down(multiplied, training=training)

        out = h + ffn_out
        return out
