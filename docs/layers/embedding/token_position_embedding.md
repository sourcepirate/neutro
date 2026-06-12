## TokenPositionEmbedding

### What does this layer do?

Combines two learnable `Embedding` lookups — one for tokens (vocabulary IDs) and one for positions (sequence indices) — and adds their outputs. This is the standard pattern used in BERT, GPT-2, and most Keras transformer implementations:

```python
x = Embedding(vocab_size, dim)(tokens) + Embedding(max_len, dim)(positions)
```

The token embedding maps each word/token to a dense vector. The position embedding maps each position `0, 1, 2, ...` to a learned vector. Adding them lets the model know both **what** token it's looking at and **where** in the sequence that token appears.

### How is this different from fixed sin/cos positional encoding?

| | Sinusoidal (`PositionalEncoding`) | Learned (`TokenPositionEmbedding`) |
|---|---|---|
| Parameters | 0 | `max_len × dim` |
| Extrapolation | Works beyond `max_len` | Can't (unseen positions get random init) |
| Training | No gradient needed | Learns position representations from data |
| Usage in this codebase | Diffusion `TimeEmbedding`, custom examples | BERT/GPT-style transformers |

Both are valid; this one matches the Keras/TF convention.

### Walking through the code

#### `__init__`

```python
class TokenPositionEmbedding(Layer):
    def __init__(self, vocab_size, max_len, dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(max_len, dim)
        self.max_len = max_len
```

🔍 **Line 1**: Inherits from `Layer` — the base class that provides `params`, `grads`, `build`, and the `sublayers` property.

🔍 **Line 3**: `vocab_size` is the number of unique tokens (e.g., 16 for the arithmetic task with digits + operators). `max_len` is the maximum sequence length (e.g., 9 tokens). `dim` is the embedding dimension (e.g., 64).

🔍 **Lines 4-5**: Two separate `Embedding` sub-layers. `token_emb` learns a `(vocab_size, dim)` table; `pos_emb` learns a `(max_len, dim)` table. They are automatically discovered by the `sublayers` property, so the optimizer iterates over both.

#### `build`

```python
def build(self, input_shape):
    self.token_emb.build(input_shape)
    self.pos_emb.build((input_shape[0], self.max_len))
    super().build(input_shape)
```

🔍 **Line 2**: Build the token embedding with the actual input shape `(batch, seq_len)`.

🔍 **Line 3**: Build the position embedding with `(batch, max_len)`. The `input_shape[0]` is the batch dimension (potentially `None`). The second dimension is `max_len` — positions are always `[0, 1, ..., max_len-1]`.

#### `forward`

```python
def forward(self, inputs, training=False):
    seq_len = inputs.shape[1]
    positions = np.arange(seq_len, dtype=np.int32).reshape(1, -1)
    return self.token_emb(inputs) + self.pos_emb(positions)
```

🔍 **Line 2**: Extract the actual sequence length from the input (could be shorter than `max_len`).

🔍 **Line 3**: Create position indices `[0, 1, 2, ..., seq_len-1]` as shape `(1, seq_len)`. The batch dimension is 1 because all samples in the batch share the same position sequence.

🔍 **Line 4**: `token_emb(inputs)` → shape `(batch, seq_len, dim)`. `pos_emb(positions)` → shape `(1, seq_len, dim)`. NumPy broadcasting repeats the position embeddings across the batch dimension. Result: `(batch, seq_len, dim)`.

📐 **Shape walkthrough**: Input `(8, 9)` with values like `[[42, 7, 3, ...], ...]` (batch of 8, sequence of 9 tokens). Token embeddings: `(8, 9, 64)`. Position embeddings: `(1, 9, 64)`. Output: `(8, 9, 64)`. Position 0 gets the same position vector for all 8 samples.

#### `backward`

```python
def backward(self, grad_output):
    self.token_emb.backward(grad_output)
    self.pos_emb.backward(grad_output.sum(axis=0, keepdims=True))
    return None
```

🔍 **Line 2**: Pass the full gradient through to `token_emb.backward`. Its `Embedding.backward` uses `np.add.at` to accumulate grad_output into the token embedding rows, indexed by the input token IDs.

🔍 **Line 3**: `pos_emb` was called with positions of shape `(1, seq_len)`, but `grad_output` has shape `(batch, seq_len, dim)`. We **sum over the batch dimension** (`axis=0`) before passing, yielding shape `(1, seq_len, dim)`.

🔍 **Why sum over batch?** All samples in the batch share the same position indices `[0, 1, 2, ...]`. The position embedding at index `i` should receive gradient contributions from position `i` in *every* sample. Summing over the batch is the mathematically correct operation — it's equivalent to having done `pos_emb(positions_broadcast_to_batch)` in the forward pass.

🔍 **`return None`**: The input tokens are discrete integer IDs (no gradient flow), and the positions are generated from `np.arange` (no gradient flow).

### Usage in a model

```python
from neutro.models import Sequential
from neutro.layers import TokenPositionEmbedding, TransformerBlock, Dense

model = Sequential([
    TokenPositionEmbedding(vocab_size=16, max_len=9, dim=64),
    TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, causal=True),
    TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, causal=True),
    Dense(16),
])
```

### Try it yourself

```python
from neutro.layers import TokenPositionEmbedding
import numpy as np

layer = TokenPositionEmbedding(vocab_size=16, max_len=9, dim=64)
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])  # (1, 9)
out = layer(x, training=True)
print(f"Output shape: {out.shape}")  # (1, 9, 64)

# Both sub-layers have their own params
print(f"Token emb params: {list(layer.token_emb.params.keys())}")  # ['embeddings']
print(f"Pos emb params:   {list(layer.pos_emb.params.keys())}")    # ['embeddings']
print(f"Token emb shape:  {layer.token_emb.params['embeddings'].shape}")  # (16, 64)
print(f"Pos emb shape:    {layer.pos_emb.params['embeddings'].shape}")    # (9, 64)
```
