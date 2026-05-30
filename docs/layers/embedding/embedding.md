# Embedding Layers

## Embedding

### What does this layer do?

Maps discrete tokens (integers like word IDs) to dense, learnable vectors. Think of it as a **lookup table**: token ID 0 gets row 0, token ID 1 gets row 1, and so on. The table entries are learned during training, so similar tokens end up with similar vectors.

### The math

$$x_i = W[\text{token}_i]$$

Where $W \in \mathbb{R}^{V \times D}$ is the embedding matrix, $V$ is the vocabulary size, and $D$ is the embedding dimension. The input is an integer tensor; the output is a float tensor where each integer has been replaced by its corresponding row of $W$.

### Walking through the code

#### `__init__` — what needs to happen before we know the shapes

```python
class Embedding(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
```

🔍 **Line 3**: `super().__init__(**kwargs)` calls `Layer.__init__`, setting up `self.params = {}`, `self.grads = {}`, and `self.built = False`. The `input_shape` kwarg is stored for later use in `build`.

🔍 **Line 4**: `input_dim` is the vocabulary size $V$ — the number of unique tokens. For example, `10000` means tokens are in range `[0, 9999]`.

🔍 **Line 5**: `output_dim` is the embedding dimension $D$ — how many numbers represent each token. Common values: `128`, `256`, `512`.

#### `build` — creating the embedding table

```python
def build(self, input_shape):
    self.params['embeddings'] = np.random.normal(0, 0.01, (self.input_dim, self.output_dim))
    super().build(input_shape)
```

🔍 **Line 2**: Create the embedding matrix. Shape `(V, D)` = `(vocab_size, embed_dim)`. Initialized with small random Gaussian values (mean 0, std 0.01).

```
        embed_dim (= 128)
    ┌──────────────────────┐
    │ token 0's embedding  │
V   │ token 1's embedding  │
o   │   ...                │
c   │                      │
a   │ token 9999's embed   │
b   └──────────────────────┘
```

🔍 **Line 3**: `super().build(input_shape)` sets `self.built = True` so `build` won't run again.

#### `forward` — looking up tokens

```python
def forward(self, inputs, training=False):
    self.inputs = inputs.astype(int)
    return self.params['embeddings'][self.inputs]
```

🔍 **Line 2**: Cast inputs to integer and cache them. We cache `self.inputs` because backward needs it to know *which rows* of the embedding table received gradients.

🔍 **Line 3**: NumPy advanced indexing: `self.params['embeddings'][self.inputs]` — for each integer in `inputs`, fetch the corresponding row of the embedding matrix.

📐 **Shape walkthrough**: Input `(B, seq_len)` with values like `[[42, 7, 999, 1]]`. `self.params['embeddings']` is `(V, D)` = `(10000, 128)`. `embeddings[inputs]` returns shape `(B, seq_len, D)` = `(1, 4, 128)`. Token 42 becomes a 128-dimensional vector (row 42 of the matrix), token 7 becomes row 7, etc.

#### `backward` — sparse gradient accumulation

```python
def backward(self, grad_output):
    self.grads['embeddings'] = np.zeros_like(self.params['embeddings'])
    np.add.at(self.grads['embeddings'], self.inputs, grad_output)
    return None
```

🔍 **Line 2**: Start with a zero gradient buffer the same shape as the embedding matrix `(V, D)`.

🔍 **Line 3**: `np.add.at(self.grads['embeddings'], self.inputs, grad_output)` — this is the critical line.

🔍 **Why `np.add.at` and not `self.grads['embeddings'][self.inputs] = ...`?** Because the same token index might appear **multiple times** in a batch. For example, if the input batch contains token 42 at two different positions, both contributions need to be **summed**, not overwritten.

Consider: `inputs = [[42, 7, 42, 1]]`, `grad_output = [[g0, g1, g2, g3]]`. Token 42 appears at positions 0 and 2. The gradient for row 42 should be `g0 + g2`. `np.add.at` handles this correctly — it accumulates into the same row.

With regular assignment `self.grads['embeddings'][self.inputs] = grad_output`, only the last occurrence of token 42 would survive (position 2's gradient `g2`), and `g0` would be silently lost.

🔍 **`return None`**: The embedding layer has no trainable parameters that affect the *previous* layer — the input tokens are fixed integers, not float gradients. There's no gradient to pass backward to a token index input. (In practice, the previous layer is usually a tokenizer or data loader, not another differentiable layer.)

---

## TimeEmbedding

### What does this layer do?

Converts a scalar timestep (a single integer like `t=42`) into a high-dimensional vector using sinusoidal functions at different frequencies. This is the positional encoding from "Attention Is All You Need", repurposed for diffusion models — it tells the model *where* in time we are.

### The math

For each timestep $t$ and embedding dimension $i$:

$$\text{TE}(t, 2i) = \sin\left(\frac{t}{10000^{2i / D}}\right)$$

$$\text{TE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i / D}}\right)$$

Where $D$ is the embedding dimension. Different dimensions oscillate at different frequencies — low-index dimensions change quickly (fine-grained time), high-index dimensions change slowly (coarse time).

### Walking through the code

#### `__init__`

```python
class TimeEmbedding(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.last_t = None
```

🔍 **Line 4**: `self.dim` is the output embedding dimension (e.g., 128 or 256). The input is a single scalar timestep; the output is a `dim`-dimensional vector.

🔍 **Line 5**: `self.last_t = None` — a cache for the timestep used in the most recent forward pass. We'll store the input timesteps here so backward can return a gradient of the same shape.

#### `forward`

```python
def forward(self, t, training=False):
    if t.ndim == 2:
        t = t.flatten()
    self.last_t = t

    half_dim = self.dim // 2
    embeddings = np.log(10000) / (half_dim - 1)
    embeddings = np.exp(np.arange(half_dim) * -embeddings)
    embeddings = t[:, None] * embeddings[None, :]
    embeddings = np.concatenate([np.sin(embeddings), np.cos(embeddings)], axis=1)

    if self.dim % 2 == 1:
        embeddings = np.pad(embeddings, ((0, 0), (0, 1)))

    return embeddings
```

🔍 **Lines 2-3**: If `t` has shape `(B, 1)`, flatten to `(B,)`. The input might come from a data loader that adds an extra dimension.

🔍 **Line 5**: `half_dim = self.dim // 2`. Since each frequency produces both a sin and a cos, we need only half as many frequencies as the total dimension.

🔍 **Lines 6-7**: Build the frequency schedule. This is the key precomputation:

```python
embeddings = np.log(10000) / (half_dim - 1)           # scalar
embeddings = np.exp(np.arange(half_dim) * -embeddings) # (half_dim,)
```

`embeddings` is a vector of frequencies. For `half_dim = 64`:
- Index 0: `exp(0 * -embeddings)` = `exp(0)` = `1.0` — highest frequency
- Index 63: `exp(63 * -embeddings)` ≈ `exp(-log(10000))` = `0.0001` — lowest frequency

These are the $1/10000^{2i/D}$ terms from the formula.

🔍 **Line 8**: Outer product: `t[:, None]` is `(B, 1)`, `embeddings[None, :]` is `(1, half_dim)`. Result: `(B, half_dim)` — each timestep multiplied by each frequency.

📐 **Shape**: `t` = `[0, 1, 2, ..., 999]` (batch of 1000 timesteps), `embeddings` = `[1.0, 0.84, ..., 0.0001]` (64 frequencies). Result: `(1000, 64)` — row `i` is `t_i * frequencies`.

🔍 **Line 9**: Apply `sin` and `cos` to get the final encoding, then concatenate along the feature axis.

📐 **Shape**: Each branch is `(B, half_dim)`. `concatenate(axis=1)` → `(B, dim)` if `dim` is even, which pairs sin(ωt) and cos(ωt) for each frequency.

🔍 **Lines 10-11**: If `dim` is odd, pad with one extra column of zeros. `np.pad(embeddings, ((0,0), (0,1)))` adds a zero column at the end.

🔍 **Why no `build` step?** The frequencies are *deterministic* — they depend only on `dim`, not on the input shape. There are no learnable parameters, so there's nothing to build.

#### `backward`

```python
def backward(self, grad_output):
    return np.zeros_like(self.last_t)
```

🔍 **Line 2**: Return a zero gradient with the same shape as the input timesteps.

🔍 **Why zeros?** The timesteps `t` are not learnable parameters — they're fixed inputs chosen by the diffusion process (e.g., `t = 0, 1, 2, ..., 999`). There's no gradient to propagate back to them. The TimeEmbedding layer is purely a feature transformation; the actual learning happens in the layers that consume its output.

🔍 **What about the gradient wrt the frequency schedule?** The frequencies are hardcoded constants (they don't appear in `self.params`), so there's no gradient to compute for them either. The `grad_output` from the next layer is simply discarded — it doesn't modify anything.

### Why sinusoidal encodings?

Sinusoidal functions have a useful property: the encoding for timestep `t + Δt` can be expressed as a linear function of the encoding for `t` (using trig identities). This means the model can learn to reason about **relative** timesteps — "50 steps from now" — rather than memorizing absolute positions.

### Try it yourself

```python
from neutro.layers import Embedding, TimeEmbedding
import numpy as np

# Token Embedding
vocab_size, embed_dim = 10000, 128
emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
tokens = np.array([[42, 7, 999, 1]])        # (1, 4)
x = emb(tokens)
print(f"Token embedding shape: {x.shape}")  # (1, 4, 128)

# Backward: check sparse accumulation
dL_dy = np.random.randn(1, 4, 128)
emb.backward(dL_dy)
print(f"Embedding grad shape: {emb.grads['embeddings'].shape}")  # (10000, 128)
print(f"Rows with non-zero gradient: {np.count_nonzero(np.any(emb.grads['embeddings'] != 0, axis=1))}")  # 3 (tokens 42, 7, 999)

# TimeEmbedding (sinusoidal positional encoding)
te = TimeEmbedding(dim=256)
timesteps = np.array([0, 1, 50, 999])       # (4,)
z = te(timesteps)
print(f"Time embedding shape: {z.shape}")   # (4, 256)
print(f"Timestep 0 encoding (first 8 dims): {z[0, :8]}")  # sin(0) = 0, cos(0) = 1 for all frequencies

# Backward returns zeros (timesteps aren't learned)
grad = te.backward(np.random.randn(4, 256))
print(f"Input gradient shape: {grad.shape}")  # (4,)
print(f"All zeros?: {np.all(grad == 0)}")     # True
```

## References

- Mikolov, T., et al. (2013). **Efficient Estimation of Word Representations in Vector Space**. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
- Vaswani, A., et al. (2017). **Attention Is All You Need**. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
