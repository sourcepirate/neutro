from ..base_model import Sequential
from ...layers.embedding.embedding import Embedding
from ...layers.normalization.rmsnorm import RMSNorm
from ...layers.core.dense import Dense
from ...layers.transformer.qwen_block import QwenBlock
from ...layers.transformer.moe_block import MoETransformerBlock


def _qwen_block(dim, n_heads, hidden_mult=3.5):
    n_kv = max(1, n_heads // 4)
    return QwenBlock(dim, n_heads, n_kv_heads=n_kv,
                     hidden_dim=int(dim * hidden_mult))

def Qwen1Tiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """
    Qwen (v1): The OG.
    Uses tied embeddings (input/output weights shared) and attention biases.
    """
    embedding = Embedding(vocab_size, dim, input_shape=(seq_len,))
    # We need to build the embedding to access its params
    embedding.build((None, seq_len))
    
    model = Sequential([embedding])
    
    for _ in range(n_layers):
        # Qwen 1 often used biases in attention
        model.add(_qwen_block(dim, n_heads))
        
    model.add(RMSNorm())
    # Tie embeddings: the output Dense layer uses the same weights as Embedding
    output_layer = Dense(vocab_size, use_bias=False)
    output_layer.build((None, seq_len, dim))
    output_layer.params['W'] = embedding.params['embeddings'].T
    model.add(output_layer)
    return model

def Qwen2Tiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """
    Qwen 2 / 2.5: Optimized for efficiency.
    Removes biases and uses Grouped Query Attention (GQA).
    """
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
    ])
    
    for _ in range(n_layers):
        # GQA is usually num_heads / 4 groups
        model.add(_qwen_block(dim, n_heads))
        
    model.add(RMSNorm())
    model.add(Dense(vocab_size, use_bias=False))
    return model

def Qwen3Tiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """Alias for a deeper ``Qwen2Tiny``; not the real Qwen3 architecture."""
    return Qwen2Tiny(vocab_size, seq_len, dim, n_layers + 2, n_heads)

def QwenTiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    # Alias for Qwen2
    return Qwen2Tiny(vocab_size, seq_len, dim, n_layers, n_heads)


def Qwen3MoETiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8,
                 n_experts=16, top_k=4):
    """Qwen3-MoE style: Qwen blocks with MoE FFN + shared expert.

    Fine-grained SwiGLU experts, top-k routing (Tiny default 16/4).
    """
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
    ])
    for _ in range(n_layers):
        model.add(MoETransformerBlock(
            dim, n_heads, n_experts=n_experts, top_k=top_k,
            expert_type='swiglu', expert_hidden_dim=int(dim * 2),
            use_shared_expert=True, use_mla=False))
    model.add(RMSNorm())
    model.add(Dense(vocab_size, use_bias=False))
    return model
