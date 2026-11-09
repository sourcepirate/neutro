from ..base_model import Sequential
from ...layers.embedding.embedding import Embedding
from ...layers.normalization.rmsnorm import RMSNorm
from ...layers.core.dense import Dense
from ...layers.transformer.qwen_block import QwenBlock
from ...layers.transformer.moe_block import MoETransformerBlock


def _qwen_block(dim, n_heads, hidden_mult=3.5):
    return QwenBlock(
        dim,
        n_heads,
        n_kv_heads=max(1, n_heads // 4),
        hidden_dim=int(dim * hidden_mult),
    )


def _qwen_blocks(model, dim, n_heads, n_layers):
    for _ in range(n_layers):
        model.add(_qwen_block(dim, n_heads))


def _qwen_model(vocab_size, seq_len, dim, n_layers, n_heads, use_bias=False):
    model = Sequential([Embedding(vocab_size, dim, input_shape=(seq_len,))])
    _qwen_blocks(model, dim, n_heads, n_layers)
    model.add(RMSNorm())
    model.add(Dense(vocab_size, use_bias=use_bias))
    return model


def Qwen1Tiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """Qwen v1 with tied input/output embeddings."""
    embedding = Embedding(vocab_size, dim, input_shape=(seq_len,))
    embedding.build((None, seq_len))

    model = Sequential([embedding])
    _qwen_blocks(model, dim, n_heads, n_layers)
    model.add(RMSNorm())

    output = Dense(vocab_size, use_bias=False)
    output.build((None, seq_len, dim))
    output.params['W'] = embedding.params['embeddings'].T
    model.add(output)
    return model


def Qwen2Tiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """Qwen 2-style stack with grouped-query attention."""
    return _qwen_model(vocab_size, seq_len, dim, n_layers, n_heads)


def Qwen3Tiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """Alias for a deeper Qwen2Tiny."""
    return Qwen2Tiny(vocab_size, seq_len, dim, n_layers + 2, n_heads)


def QwenTiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """Alias for Qwen2Tiny."""
    return Qwen2Tiny(vocab_size, seq_len, dim, n_layers, n_heads)


def Qwen3MoETiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8,
                 n_experts=16, top_k=4):
    """Qwen-style blocks with a shared-expert MoE feed-forward."""
    model = Sequential([Embedding(vocab_size, dim, input_shape=(seq_len,))])
    for _ in range(n_layers):
        model.add(MoETransformerBlock(
            dim, n_heads, n_experts=n_experts, top_k=top_k,
            expert_type='swiglu', expert_hidden_dim=int(dim * 2),
            use_shared_expert=True, use_mla=False,
        ))
    model.add(RMSNorm())
    model.add(Dense(vocab_size, use_bias=False))
    return model
