import numpy as np
from ..base_model import Sequential
from ...layers.embedding.embedding import Embedding
from ...layers.normalization.rmsnorm import RMSNorm
from ...layers.core.dense import Dense
from ...layers.transformer.bitnet_block import BitNetBlock


def BitNetTiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8, mode='b1.58', activation_bits=8):
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
    ])

    for _ in range(n_layers):
        model.add(BitNetBlock(
            embed_dim=dim,
            num_heads=n_heads,
            ff_dim=dim * 4,
            mode=mode,
            activation_bits=activation_bits,
        ))

    model.add(RMSNorm())
    model.add(Dense(vocab_size, use_bias=False))
    return model
