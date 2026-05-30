import numpy as np
from ..base import Layer
from ..attention.mha import MultiHeadAttention
from ..attention.flash_attention import FlashAttention
from ..attention.base_attention import BaseAttention
from ..core.dense import Dense
from ..core.dropout import Dropout
from ..normalization.layernorm import LayerNormalization

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, causal=False, use_flash=False, pre_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.causal = causal
        self.pre_norm = pre_norm
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

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, inputs, training=False, kv_cache=None, layer_id=None):
        self.inputs = inputs
        mask = None
        if self.causal:
            mask = BaseAttention.create_causal_mask(inputs.shape[1])
            if kv_cache and layer_id in kv_cache.k_cache:
                # If using cache, mask needs to account for previous tokens
                q_len = inputs.shape[1]
                kv_len = q_len + kv_cache.k_cache[layer_id].shape[2]
                mask = np.ones((q_len, kv_len))
                # Only the triangular part for the current tokens
                mask[:, -q_len:] = BaseAttention.create_causal_mask(q_len)
                # Past tokens are always visible
                mask[:, :-q_len] = 0
            
        if self.pre_norm:
            # Pre-Norm architecture
            norm1 = self.layernorm1(inputs, training=training)
            attn_output = self.att(norm1, mask=mask, training=training, kv_cache=kv_cache, layer_id=layer_id)
            attn_dropped = self.dropout1(attn_output, training=training)
            h = inputs + attn_dropped
            
            norm2 = self.layernorm2(h, training=training)
            ffn_1 = self.ffn[0](norm2, training=training)
            ffn_2 = self.ffn[1](ffn_1, training=training)
            ffn_dropped = self.dropout2(ffn_2, training=training)
            return h + ffn_dropped
        else:
            # Post-Norm architecture
            attn_output = self.att(inputs, mask=mask, training=training, kv_cache=kv_cache, layer_id=layer_id)
            attn_dropped = self.dropout1(attn_output, training=training)
            self.out1_pre_ln = inputs + attn_dropped
            out1 = self.layernorm1(self.out1_pre_ln, training=training)
            
            ffn_1 = self.ffn[0](out1, training=training)
            ffn_2 = self.ffn[1](ffn_1, training=training)
            ffn_dropped = self.dropout2(ffn_2, training=training)
            self.out2_pre_ln = out1 + ffn_dropped
            return self.layernorm2(self.out2_pre_ln, training=training)

    def backward(self, grad_output):
        if self.pre_norm:
            # Pre-Norm backward
            # out = h + dropout2(ffn2(ffn1(norm2(h))))
            grad_ffn_path = self.dropout2.backward(grad_output)
            grad_ffn = self.ffn[1].backward(grad_ffn_path)
            grad_ffn = self.ffn[0].backward(grad_ffn)
            grad_norm2 = self.layernorm2.backward(grad_ffn)
            grad_h = grad_output + grad_norm2
            
            # h = inputs + dropout1(attn(norm1(inputs)))
            grad_attn_path = self.dropout1.backward(grad_h)
            grad_attn = self.att.backward(grad_attn_path)
            grad_norm1 = self.layernorm1.backward(grad_attn)
            return grad_h + grad_norm1
        else:
            # Post-Norm backward
            grad = self.layernorm2.backward(grad_output)
            grad_ffn_path = self.dropout2.backward(grad)
            grad_ffn = self.ffn[1].backward(grad_ffn_path)
            grad_ffn = self.ffn[0].backward(grad_ffn)
            grad_out1 = self.layernorm1.backward(grad + grad_ffn)
            
            grad_attn_path = self.dropout1.backward(grad_out1)
            grad_attn = self.att.backward(grad_attn_path)
            return grad_out1 + grad_attn
