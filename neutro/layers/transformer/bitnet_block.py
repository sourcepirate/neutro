import numpy as np
from ..base import Layer
from ..core.bitlinear import BitLinear
from ..normalization.rmsnorm import RMSNorm


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
        self.scale = 1.0 / np.sqrt(self.head_dim)

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
        self.inputs = inputs
        B, S, D = inputs.shape
        H = self.num_heads
        Dh = self.head_dim

        h1_norm = self.attn_norm(inputs, training=training)
        self.h1_norm = h1_norm

        Q = self.wq(h1_norm, training=training)
        K = self.wk(h1_norm, training=training)
        V = self.wv(h1_norm, training=training)

        Q = Q.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        K = K.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores += mask * -1e9

        m = np.max(scores, axis=-1, keepdims=True)
        p = np.exp(scores - m)
        l = np.sum(p, axis=-1, keepdims=True) + 1e-15
        attn_weights = p / l
        self.attn_weights = attn_weights
        self.Q = Q
        self.K = K
        self.V = V

        attn_out = attn_weights @ V
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
        self.attn_out = attn_out

        attn_proj = self.wo(attn_out, training=training)
        self.attn_proj = attn_proj

        self.h = inputs + attn_proj

        h2_norm = self.ffn_norm(self.h, training=training)
        self.h2_norm = h2_norm

        gate = self.ffn_gate(h2_norm, training=training)
        up = self.ffn_up(h2_norm, training=training)

        sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
        activated_gate = gate * sigmoid_gate
        self.gate = gate
        self.sigmoid_gate = sigmoid_gate
        self.up = up
        self.activated_gate = activated_gate
        self.multiplied = activated_gate * up

        ffn_out = self.ffn_down(self.multiplied, training=training)
        self.ffn_out = ffn_out

        out = self.h + ffn_out
        return out

    def backward(self, grad_output):
        grad_h2 = grad_output

        grad_ffn_down = self.ffn_down.backward(grad_h2)
        grad_multiplied = grad_ffn_down

        grad_activated_gate = grad_multiplied * self.up
        grad_up = grad_multiplied * self.activated_gate

        dsigmoid = self.sigmoid_gate * (1.0 - self.sigmoid_gate)
        grad_gate = grad_activated_gate * (self.gate * dsigmoid + self.sigmoid_gate)

        grad_h2_ffn = self.ffn_up.backward(grad_up)
        grad_h2_gate = self.ffn_gate.backward(grad_gate)
        grad_ffn_norm = self.ffn_norm.backward(grad_h2_ffn + grad_h2_gate)

        grad_h = grad_output + grad_ffn_norm

        grad_attn_proj = self.wo.backward(grad_h)
        grad_attn_out = grad_attn_proj

        B, S, D = self.inputs.shape
        H = self.num_heads
        Dh = self.head_dim
        grad_attn_out = grad_attn_out.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

        dV = self.attn_weights.transpose(0, 1, 3, 2) @ grad_attn_out
        dS = grad_attn_out @ self.V.transpose(0, 1, 3, 2)
        dP = dS * self.scale
        dsmax = self.attn_weights * (dP - np.sum(dP * self.attn_weights, axis=-1, keepdims=True))
        dQ = dsmax @ self.K
        dK = dsmax.transpose(0, 1, 3, 2) @ self.Q

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, S, D)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, S, D)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, S, D)

        grad_h1_norm = self.wq.backward(dQ) + self.wk.backward(dK) + self.wv.backward(dV)
        grad_attn_norm = self.attn_norm.backward(grad_h1_norm)

        return grad_h + grad_attn_norm
