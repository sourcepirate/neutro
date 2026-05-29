from ..base import Layer
from ..attention.mha import MultiHeadAttention
from ..attention.flash_attention import FlashAttention
from ..core.dense import Dense
from ..core.dropout import Dropout
from ..normalization.layernorm import LayerNormalization

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, causal=False, use_flash=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.causal = causal
        if use_flash:
            self.att = FlashAttention(num_heads=num_heads, key_dim=embed_dim)
        else:
            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def build(self, input_shape):
        self.att.build(input_shape)
        curr_shape = input_shape
        for layer in self.ffn:
            layer.build(curr_shape)
            curr_shape = (input_shape[0], input_shape[1], layer.units)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.dropout1.build(input_shape)
        self.dropout2.build(input_shape)
        super().build(input_shape)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        mask = None
        if self.causal:
            from ..attention.base_attention import BaseAttention
            mask = BaseAttention.create_causal_mask(inputs.shape[1])
            
        attn_output = self.att(inputs, mask=mask, training=training)
        self.attn_output = attn_output
        attn_dropped = self.dropout1(attn_output, training=training)
        self.out1_pre_ln = inputs + attn_dropped
        out1 = self.layernorm1(self.out1_pre_ln)
        self.out1 = out1
        ffn_1 = self.ffn[0](out1, training=training)
        ffn_2 = self.ffn[1](ffn_1, training=training)
        self.ffn_output = ffn_2
        ffn_dropped = self.dropout2(ffn_2, training=training)
        self.out2_pre_ln = out1 + ffn_dropped
        return self.layernorm2(self.out2_pre_ln)

    def backward(self, grad_output):
        grad = self.layernorm2.backward(grad_output)
        grad_ffn_path = self.dropout2.backward(grad)
        grad_skip_out1 = grad
        grad_ffn = self.ffn[1].backward(grad_ffn_path)
        grad_ffn = self.ffn[0].backward(grad_ffn)
        grad_out1 = self.layernorm1.backward(grad_skip_out1 + grad_ffn)
        grad_attn_path = self.dropout1.backward(grad_out1)
        grad_skip_inputs = grad_out1
        grad_attn = self.att.backward(grad_attn_path)
        if grad_attn is None: return grad_skip_inputs
        return grad_skip_inputs + grad_attn
