# FlashAttention

## What does this layer do?

FlashAttention computes **exact** scaled dot-product attention, but it **never materializes the full $N \times N$ attention matrix** in memory. Instead, it processes the input in **tiles** (blocks) using a clever **online softmax** algorithm. This reduces memory from $O(N^2)$ to $O(N)$ — a huge deal for long sequences (4K, 8K, 128K tokens).

**Important**: This is NOT an approximation. The output is **bitwise identical** to standard attention. The only savings are in memory bandwidth and peak memory usage.

## The math, in plain English

### Standard attention memory problem

Standard attention: $$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

The intermediate matrix $S = QK^T$ has shape $(N, N)$, where $N$ is the sequence length. For $N=128K$, that's $128K^2 \approx 16\text{ billion}$ floats — 64 GB for a single attention layer!

### The tiling trick

Instead of computing the full $S$, FlashAttention processes small blocks:

1. **Outer loop** (over columns of K, V): Load a tile of K and V.
2. **Inner loop** (over rows of Q): Load a tile of Q.
3. For each tile pair $(Q_i, K_j)$, compute the local attention scores $S_{ij} = Q_i K_j^T / \sqrt{d}$.
4. **Combine** with the running statistics using **online softmax** to get the correct partial output.

### Online softmax

For each row $r$, we maintain:
- **$M_r$**: The maximum score seen so far across all tiles.
- **$L_r$**: The sum of exponentials (normalized by the max) seen so far.
- **$O_r$**: The partial output accumulated so far.

When processing a new tile $j$:

1. Compute local max $m_{ij}$ and local sum $l_{ij}$ of $P_{ij} = e^{S_{ij} - m_{ij}}$.
2. Get the **new global max**: $M_r^{\text{new}} = \max(M_r, m_{ij})$.
3. Rescale the old statistics:
   - $\alpha = e^{M_r - M_r^{\text{new}}}$
   - $\beta = e^{m_{ij} - M_r^{\text{new}}}$
4. Update running values:
   - $O_r = \alpha \cdot O_r + \beta \cdot (P_{ij} \cdot V_j)$
   - $L_r = \alpha \cdot L_r + \beta \cdot l_{ij}$
   - $M_r = M_r^{\text{new}}$

After all tiles: $O = O / L$ (final normalization).

---

## Walking through the code

### File: `neutro/layers/attention/flash_attention.py`

### Step 1: `__init__` — line 18

```python
class FlashAttention(Layer):
    def __init__(self, num_heads, key_dim, block_size_r=64, block_size_c=64, dropout=0.0, use_rope=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
        self.block_size_r = block_size_r
        self.block_size_c = block_size_c
        self.dropout_rate = dropout
        self.use_rope = use_rope
        self.scale = 1.0 / np.sqrt(self.head_dim)
```

🔍 **Line 18**: `FlashAttention` inherits from `Layer` directly. It doesn't need `BaseAttention` because it implements its own tiled attention.

🔍 **Line 23**: `block_size_r` — the tile size for **rows** of Q. Controls how many query positions we process at once.

🔍 **Line 24**: `block_size_c` — the tile size for **columns** of K and V. Controls how many key/value positions we load at once.

🔍 **Line 26**: `use_rope` — whether to apply Rotary Position Embeddings before attention.

### Step 2: `build` — line 29

```python
def build(self, input_shape):
    self.embed_dim = input_shape[-1]
    self.params['Wq'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
    self.params['Wk'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
    self.params['Wv'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
    self.params['Wo'] = np.random.randn(self.key_dim, self.embed_dim) * 0.02
    super().build(input_shape)
```

🔍 **Lines 32–35**: Standard MHA-style projections: four weight matrices. The initialization uses `randn * 0.02` (small random values) instead of Glorot — this is a common choice for Transformers (GPT-2 style).

### Step 3: `forward` — line 41

#### Initial projections and head splitting

```python
def forward(self, x, mask=None, training=False, kv_cache=None, layer_id=None):
    self.x = x
    self.mask = mask
    batch_size, seq_len, _ = x.shape
    H = self.num_heads
    d = self.head_dim
    K_dim = self.key_dim

    Q = (x @ self.params['Wq']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
    K = (x @ self.params['Wk']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
    V = (x @ self.params['Wv']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
```

🔍 **Lines 49–52**: Standard Q, K, V projections and head splitting. Shapes go from `(B, S, D)` to `(B, H, S, d)`.

#### RoPE (optional)

```python
    if self.use_rope:
        total_seq_len = seq_len
        if kv_cache and layer_id in kv_cache.k_cache:
            total_seq_len += kv_cache.k_cache[layer_id].shape[2]
        
        self.freqs_cis = precompute_freqs_cis(self.head_dim, total_seq_len)
        if seq_len == 1 and total_seq_len > 1:
            f_cis = self.freqs_cis[total_seq_len-1:total_seq_len]
        else:
            f_cis = self.freqs_cis[:seq_len]
            
        Q = apply_rotary_emb(Q, f_cis)
        K = apply_rotary_emb(K, f_cis)
```

🔍 **Lines 54–69**: If RoPE is enabled, precompute the frequency cisoids for the total sequence length (including cached tokens), then apply rotary embeddings to Q and K. During generation (`seq_len=1` with cache), only the last position's RoPE is applied.

#### KV cache

```python
    if kv_cache is not None and layer_id is not None:
        K, V = kv_cache.update(K, V, layer_id)
        seq_len_kv = K.shape[2]
    else:
        seq_len_kv = seq_len
```

🔍 **Lines 72–77**: Standard KV cache update. `seq_len_kv` may now be larger than `seq_len` (when generating with cache).

```python
    self.Q, self.K, self.V = Q, K, V
```

🔍 **Line 79**: Cache Q, K, V for the backward pass.

#### Initialize output and running statistics

```python
    O = np.zeros_like(Q)
    L = np.zeros((batch_size, H, seq_len, 1))
    M = np.full((batch_size, H, seq_len, 1), -np.inf)
```

🔍 **Line 83**: `O` — the running output, starts as all zeros.

🔍 **Line 84**: `L` — the running sum of exponentials, starts at 0.

🔍 **Line 85**: `M` — the running max per row, starts at $-\infty$ so the first tile's max always wins.

All three are `(B, H, S_q, 1)` — one value per query position per head.

#### Tiling setup

```python
    Br = self.block_size_r
    Bc = self.block_size_c
    Tr = (seq_len + Br - 1) // Br
    Tc = (seq_len_kv + Bc - 1) // Bc
```

🔍 **Lines 88–91**: `Tr` and `Tc` are the **number of tiles** along the query and key dimensions. The formula `(N + block - 1) // block` is ceiling division — ensures we cover all positions even if `N` isn't a multiple of `block`.

#### Outer loop: tiles of K, V

```python
    for j in range(Tc):
        j_start, j_end = j * Bc, min((j + 1) * Bc, seq_len_kv)
        Kj = K[:, :, j_start:j_end, :]  # (batch, H, Bc, d)
        Vj = V[:, :, j_start:j_end, :]  # (batch, H, Bc, d)
```

🔍 **Lines 93–96**: **Outer loop** — iterate over columns of K and V. Each tile `Kj` has shape `(B, H, Bc, d)`.

#### Inner loop: tiles of Q

```python
        for i in range(Tr):
            i_start, i_end = i * Br, min((i + 1) * Br, seq_len)
            Qi = Q[:, :, i_start:i_end, :]  # (batch, H, Br, d)
            Oi = O[:, :, i_start:i_end, :]
            Mi = M[:, :, i_start:i_end, :]
            Li = L[:, :, i_start:i_end, :]
```

🔍 **Lines 98–103**: **Inner loop** — iterate over rows of Q. Each tile `Qi` has shape `(B, H, Br, d)`. We also slice the corresponding portions of O, M, and L.

#### Compute attention scores for this tile

```python
            S_ij = self.scale * (Qi @ Kj.transpose(0, 1, 3, 2))  # (batch, H, Br, Bc)
            
            if mask is not None:
                m_tile = mask[i_start:i_end, j_start:j_end]
                S_ij -= 1e9 * m_tile
```

🔍 **Line 106**: **Local attention scores** for this tile pair. Shape `(B, H, Br, Bc)` — note this is **much** smaller than the full `(B, H, S_q, S_kv)` matrix!

📐 `(B, H, Br, d) @ (B, H, d, Bc)` → `(B, H, Br, Bc)`

🔍 **Lines 108–111**: Apply the mask to this tile. If the mask is causal, only the relevant portion of the causal mask is applied.

#### Online softmax update

```python
            m_ij = np.max(S_ij, axis=-1, keepdims=True)
            P_ij = np.exp(S_ij - m_ij)
            l_ij = np.sum(P_ij, axis=-1, keepdims=True)

            M_new = np.maximum(Mi, m_ij)
            
            alpha = np.exp(Mi - M_new)
            beta = np.exp(m_ij - M_new)
            
            O[:, :, i_start:i_end, :] = alpha * Oi + beta * (P_ij @ Vj)
            M[:, :, i_start:i_end, :] = M_new
            L[:, :, i_start:i_end, :] = alpha * Li + beta * l_ij
```

🔍 **Line 114**: **Local max** `m_ij` within this tile — `(B, H, Br, 1)`.

🔍 **Line 115**: **Local exponentiated scores** `P_ij` — `(B, H, Br, Bc)`. Only this tile's worth of the softmax numerator is computed.

🔍 **Line 116**: **Local sum** `l_ij` — `(B, H, Br, 1)`.

🔍 **Line 118**: **Updated global max** — element-wise max of the previous max `Mi` and the new tile's max `m_ij`.

🔍 **Line 121**: **Rescaling factor** for the old statistics: $\alpha = e^{M_i - M_{\text{new}}}$. If the previous max was lower, $\alpha < 1$, downweighting the old contributions.

🔍 **Line 122**: **Rescaling factor** for the new tile: $\beta = e^{m_{ij} - M_{\text{new}}}$. If this tile's max equals the new global max, $\beta = 1$. If it's lower, $\beta < 1$.

🔍 **Line 125**: **Update output**: blend old output and new tile's output, each rescaled by the correct factor.

`P_ij @ Vj` has shape `(B, H, Br, Bc) @ (B, H, Bc, d)` → `(B, H, Br, d)` — the contribution from this tile.

🔍 **Line 127**: **Update running sum** `L` with the same rescaling logic.

#### Final normalization

```python
    O = O / L
    self.O_pre_proj = O
    self.L = L
    self.M = M
    
    O_merged = O.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
    return O_merged @ self.params['Wo']
```

🔍 **Line 130**: **Final normalization**: divide each row's output by its running sum $L$. This gives the **correct** attention output, identical to what standard softmax would produce.

🔍 **Lines 131–133**: Cache `O`, `L`, `M` for the backward pass.

🔍 **Lines 136–137**: Merge heads and apply output projection.

### Step 4: `backward` — line 139

```python
def backward(self, grad_output):
    batch_size, seq_len, embed_dim = self.x.shape
    H = self.num_heads
    d = self.head_dim
    K_dim = self.key_dim
    
    # dWo
    O_merged = self.O_pre_proj.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
    self.grads['Wo'] = O_merged.reshape(-1, K_dim).T @ grad_output.reshape(-1, embed_dim)
    
    # dO_pre_proj
    do_merged = grad_output @ self.params['Wo'].T
    do = do_merged.reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
```

🔍 **Lines 145–151**: Standard output projection backward — same as MHA.

```python
    # D = rowsum(dO * O)
    D = np.sum(do * self.O_pre_proj, axis=-1, keepdims=True)
```

🔍 **Line 154**: **The key trick in FlashAttention backward!** $D$ is the row-wise sum of $dO \odot O$, which is used to recompute the attention gradients without storing the full attention matrix.

#### Re-tile the backward pass

```python
    dQ = np.zeros_like(self.Q)
    dK = np.zeros_like(self.K)
    dV = np.zeros_like(self.V)
    
    Br = self.block_size_r
    Bc = self.block_size_c
    Tr = (seq_len + Br - 1) // Br
    Tc = (seq_len + Bc - 1) // Bc
    
    for j in range(Tc):
        j_start, j_end = j * Bc, min((j + 1) * Bc, seq_len)
        Kj = self.K[:, :, j_start:j_end, :]
        Vj = self.V[:, :, j_start:j_end, :]
        
        dkj = np.zeros_like(Kj)
        dvj = np.zeros_like(Vj)
        
        for i in range(Tr):
            i_start, i_end = i * Br, min((i + 1) * Br, seq_len)
            Qi = self.Q[:, :, i_start:i_end, :]
            doi = do[:, :, i_start:i_end, :]
            Mi = self.M[:, :, i_start:i_end, :]
            Li = self.L[:, :, i_start:i_end, :]
            Di = D[:, :, i_start:i_end, :]
```

🔍 **Lines 156–179**: Same tiling structure as forward. We **recompute** the attention scores from `M` and `L` (which we cached) instead of storing the full attention matrix.

```python
            # Recompute A_ij = exp(S_ij - M_i) / L_i
            S_ij = self.scale * (Qi @ Kj.transpose(0, 1, 3, 2))
            if self.mask is not None:
                m_tile = self.mask[i_start:i_end, j_start:j_end]
                S_ij -= 1e9 * m_tile
            
            A_ij = np.exp(S_ij - Mi) / Li
```

🔍 **Lines 182–187**: **Recompute the normalized attention weights** for this tile using the cached `M` and `L`. This means we compute $A_{ij} = e^{S_{ij} - M_i} / L_i$ (where $M_i$ is the **global** max, not the local one). This gives the **correct** softmax value for this tile, accounting for contributions from all other tiles.

```python
            dvj += A_ij.transpose(0, 1, 3, 2) @ doi
            dS_ij = A_ij * (doi @ Vj.transpose(0, 1, 3, 2) - Di)
            
            dQ[:, :, i_start:i_end, :] += self.scale * (dS_ij @ Kj)
            dkj += self.scale * (dS_ij.transpose(0, 1, 3, 2) @ Qi)
        
        dK[:, :, j_start:j_end, :] = dkj
        dV[:, :, j_start:j_end, :] = dvj
```

🔍 **Lines 189–196**: **Attention gradient computation** for this tile:

- **`dvj`**: $dV_j = \sum_i A_{ij}^T \cdot dO_i$ — accumulate gradient for V from all Q tiles.
- **`dS_ij`**: $dS_{ij} = A_{ij} \odot (dO_i \cdot V_j^T - D_i)$ — the softmax gradient for this tile.
- **`dQi`**: $dQ_i = \sum_j dS_{ij} \cdot K_j / \sqrt{d}$ — accumulate gradient for Q from all K tiles.
- **`dkj`**: $dK_j = \sum_i dS_{ij}^T \cdot Q_i / \sqrt{d}$ — accumulate gradient for K from all Q tiles.

```python
    if self.use_rope:
        dQ = apply_rotary_emb(dQ, np.conj(self.freqs_cis))
        dK = apply_rotary_emb(dK, np.conj(self.freqs_cis))
```

🔍 **Lines 198–200**: **Reverse RoPE**: apply the complex conjugate of the rotation to get gradient in the original space.

```python
    # Map back to weights
    dq_flat = dQ.transpose(0, 2, 1, 3).reshape(-1, K_dim)
    dk_flat = dK.transpose(0, 2, 1, 3).reshape(-1, K_dim)
    dv_flat = dV.transpose(0, 2, 1, 3).reshape(-1, K_dim)
    x_flat = self.x.reshape(-1, embed_dim)
    
    self.grads['Wq'] = x_flat.T @ dq_flat
    self.grads['Wk'] = x_flat.T @ dk_flat
    self.grads['Wv'] = x_flat.T @ dv_flat
    
    return (dq_flat @ self.params['Wq'].T + dk_flat @ self.params['Wk'].T + dv_flat @ self.params['Wv'].T).reshape(batch_size, seq_len, embed_dim)
```

🔍 **Lines 202–212**: Same projection gradient computation as MHA: flatten, compute `dW = x^T @ d_proj`, and sum the input gradients from Q, K, V paths.

## Why this is exact, not approximate

The key insight: at the end of the forward pass, we divide by `L` which has accumulated the **correct** sum of exponentials (because the rescaling factors $\alpha$ and $\beta$ account for the changing max). And in the backward pass, we recompute the attention weights using the **final** max `M` and sum `L`, not the intermediate ones.

Every step is mathematically equivalent to standard attention — the only difference is **when** and **in what order** the arithmetic is performed.

## Usage Example

```python
from neutro.layers.attention import FlashAttention
import numpy as np

layer = FlashAttention(num_heads=8, key_dim=512, block_size_r=32, block_size_c=32)
x = np.random.randn(4, 128, 256)
layer.build(x.shape)
y = layer(x)  # Uses tiled attention under the hood
```

## References

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. *NeurIPS*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, T. (2023). **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**. *arXiv:2307.08691*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
