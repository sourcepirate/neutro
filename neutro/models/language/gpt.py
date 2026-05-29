from ..base_model import Sequential
from ...layers import Embedding, TransformerBlock, Dense, Softmax
from ...layers.normalization.layernorm import LayerNormalization

def GPT1(vocab_size, seq_len, dim=768, n_heads=12, n_layers=12):
    """
    GPT-1: The beginning of the end for our free time.
    Uses learned positional embeddings and "Post-Norm" blocks.
    """
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
        # Note: In a true GPT, we'd add PositionalEmbeddings here.
        # For our naive version, Embedding handles the tokens.
    ])
    
    for _ in range(n_layers):
        # GPT-1 was a bit simpler
        model.add(TransformerBlock(dim, n_heads, ff_dim=4*dim, causal=True))
        
    model.add(Dense(vocab_size))
    model.add(Softmax())
    return model

def GPT2(vocab_size, seq_len, dim=768, n_heads=12, n_layers=12):
    """
    GPT-2: The one that could write unicorns.
    The main change was moving LayerNorm to "Pre-Norm" for better stability.
    """
    model = Sequential([
        Embedding(vocab_size, dim, input_shape=(seq_len,)),
    ])
    
    for _ in range(n_layers):
        model.add(TransformerBlock(dim, n_heads, ff_dim=4*dim, causal=True, pre_norm=True))
        
    model.add(LayerNormalization())
    model.add(Dense(vocab_size))
    model.add(Softmax())
    return model
