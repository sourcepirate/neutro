from ..base_model import Sequential
from ...layers.embedding.embedding import Embedding
from ...layers.normalization.rmsnorm import RMSNorm
from ...layers.core.dense import Dense
from .llama import LlamaBlock

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
        model.add(LlamaBlock(dim, n_heads, n_heads, dim // n_heads, int(dim * 3.5)))
        
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
        n_kv_heads = max(1, n_heads // 4)
        model.add(LlamaBlock(dim, n_heads, n_kv_heads, dim // n_heads, int(dim * 3.5)))
        
    model.add(RMSNorm())
    model.add(Dense(vocab_size, use_bias=False))
    return model

def Qwen3Tiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    """
    Qwen 3 (Preview/Speculative):
    Refined architecture, likely building on Qwen2.5's strengths.
    """
    return Qwen2Tiny(vocab_size, seq_len, dim, n_layers + 2, n_heads)

def QwenTiny(vocab_size, seq_len, dim=512, n_layers=4, n_heads=8):
    # Alias for Qwen2
    return Qwen2Tiny(vocab_size, seq_len, dim, n_layers, n_heads)
