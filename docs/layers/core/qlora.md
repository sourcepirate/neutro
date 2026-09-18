# QLoRADense Layer

## What does this layer do?

QLoRA (Quantized LoRA, Dettmers et al. 2023) combines **4-bit quantization**
of the frozen base weight with a LoRA adapter. The base weight is stored as
4-bit NormalFloat (NF4) codes and dequantized on-the-fly:

```
y = x @ dequant(W_nf4) + b_frozen + s * (x @ A) @ B
```

- `W_nf4` — the frozen `D × U` weight quantized to 4-bit NF4, block-wise
  absmax-normalized. Uses ~0.5 bytes/param instead of 4–8 bytes.
- `A (D, r)` and `B (r, U)` — the same trainable LoRA matrices as in
  `LoRADense`. `s = alpha / r`.
- **Double quantization** (optional): the per-block absmax scales are
  themselves quantized to 8-bit per super-block, saving another ~0.3 bits/param.

Result: you can fine-tune a 7B-parameter model's worth of frozen weights in
~3.5 GB (NF4) plus a few MB of LoRA, instead of 28 GB in FP32.

## The math, in plain English

### NF4 quantization

NF4 is a 4-bit codebook with 16 levels **optimized for normally-distributed
weights** (quantile quantization). The codebook values are:

```
NF4 = [-1.00, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.00,
        0.080,  0.161,  0.246,  0.338,  0.441,  0.563,  0.723, 1.00]
```

Block-wise absmax:

```
For each contiguous block of `block_size` (=64) weights:
    absmax_b = max(|W_block|)                  — float scale for this block
    W_norm   = W_block / absmax_b             — normalized to [-1, 1]
    code_b   = argmin_k |W_norm - NF4[k]|     — 4-bit index (0–15)
```

Dequantization:

```
W_block ≈ NF4[code_b] * absmax_b
```

### Double quantization

```
absmax is shape (num_blocks,).  Quantize it again:
    super_block of 256 absmax values:
        s2 = max(absmax_super)
        q_absmax = round(absmax / s2 * 255)   — uint8
        absmax ≈ q_absmax / 255 * s2
```

Overhead per param:
- Without double quant: `1 float / 64` = 0.5 bits/param.
- With double quant: `1 byte / 64 + 1 float / (64·256)` ≈ 0.13 bits/param.

### QLoRA forward

```
W_dq     = dequant(W_nf4, absmax)                 (D, U) float
h        = x @ A                                  (B, r)
lora     = h @ B * s                              (B, U)
y        = x @ W_dq + b + lora                    (B, U)
y        = phi(y)  if activation
```

Gradients are **identical to LoRA** — `W_dq` is treated as a frozen constant,
so only `A, B` receive gradients. Dequantization is not differentiated through.

## Walking through the code

File: `neutro/layers/core/qlora.py`
Quantization file: `neutro/utils/quantization.py`

### Step 1: `__init__` — same as LoRA plus quantization config

```python
class QLoRADense(Layer):
    def __init__(self, units, rank=8, alpha=16, dropout=0.0,
                 block_size=64, double_quant=True,
                 double_quant_block_size=256, ...):
        self.block_size = int(block_size)               # NF4 block size
        self.double_quant = bool(double_quant)
        self.double_quant_block_size = int(double_quant_block_size)
        self.W_codes = None       # uint8 NF4 indices (D, U) 🔍 frozen
        self.W_absmax = None      # float or uint8 per-block absmax 🔍
        self.W_absmax_dq = None   # dequantized absmax when double_quant 🔍
        self.W_scales2 = None     # second-level scales 🔍
        self.W_shape = None       # original (D, U) 🔍
```

🔍 **`block_size`** (line ~30): Number of weights sharing one absmax.
Smaller blocks → lower quantization error but more overhead. 64 is the
QLoRA default (good trade-off).

🔍 **`double_quant`** (line ~31): If `True`, the absmax array itself is
quantized to int8 per super-block of 256. Saves ~0.37 bits/param.

🔍 **`W_codes`** (line ~36): `uint8` array `(D, U)` values 0–15 — the NF4
indices. This is the compressed storage; ~0.5 bytes/param.

🔍 **`W_absmax` / `W_absmax_dq` / `W_scales2`** (lines ~37–39): When
`double_quant=True`, `W_absmax` holds `uint8` codes, `W_absmax_dq` holds
the dequantized float scales used in `forward`, and `W_scales2` holds the
second-level float scales. When `False`, `W_absmax` is the float absmax
directly.

### Step 2: `build` — initialize, then quantize

```python
def build(self, input_shape):
    self.input_dim = input_shape[-1]
    W = self.kernel_initializer((self.input_dim, self.units)).astype(float)  # (D, U)
    self.W_shape = W.shape
    codes, absmax, _ = quantize_nf4(W, block_size=self.block_size)           # 🔍
    self.W_codes = codes
    if self.double_quant:
        q_absmax, dq_absmax, scales2, _ = quantize_absmax(absmax, block_size=256)
        self.W_absmax = q_absmax       # uint8
        self.W_absmax_dq = dq_absmax   # float (cached dequant)
        self.W_scales2 = scales2
    else:
        self.W_absmax = absmax         # float
    self.params['lora_A'] = Tensor(self.lora_initializer((self.input_dim, self.rank)))
    self.params['lora_B'] = Tensor(np.zeros((self.rank, self.units)))
    super().build(input_shape)
```

🔍 **`quantize_nf4`** (line ~80): `neutro/utils/quantization.py:30`.
Block-wise absmax + nearest-NF4-codebook lookup. Returns
`(codes, absmax, shape)`. See `quantization.py` walkthrough below.

🔍 **`quantize_absmax`** (line ~84): `quantization.py:115`. Second-level
8-bit uniform quantization of the absmax array (256 absmax values per
`scales2` entry). Returns `(q_codes, dq_float, scales2, shape)`.

📐 **Shape walkthrough**: For `D=32, U=64`, `block_size=64`: total
`32·64=2048` weights → `2048/64=32` blocks → `absmax` shape `(32,)`.
With `double_quant` block 256: `32` absmax values → 1 super-block
(since 32 < 256, no padding effect in practice).

### Step 3: `_dequant_weight` — reconstruct float weight

```python
def _dequant_weight(self):
    if self.double_quant:
        absmax = dequantize_absmax(self.W_absmax, self.W_scales2,
                                   block_size=self.double_quant_block_size,
                                   shape=self._absmax_shape)  # float (num_blocks,)
    else:
        absmax = self.W_absmax
    W = dequantize_nf4(self.W_codes, absmax,
                       block_size=self.block_size, shape=self.W_shape)  # (D, U)
    return W
```

🔍 **Line ~106–128**: Every `forward` call dequantizes `W` to float.
This is the cost QLoRA pays for memory savings — one extra
codebook lookup + multiply per weight. For the `neutro` educational
scale this is negligible.

### Step 4: `forward` — dequant + LoRA branch

```python
def forward(self, inputs, training=False):
    x = inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs)
    self._cache_x = x                          # 🔍 for backward
    x_flat = x.reshape(-1, self.input_dim)
    # dropout on LoRA input (same as LoRADense)
    x_lora = x_flat * mask if training and dropout else x_flat
    self._cache_x_lora = x_lora                # 🔍 for grad_A
    W_dq = self._dequant_weight()              # (D, U) 🔍 dequantized, for dx
    self._cache_W_dq = W_dq
    A = _get_data(self.params['lora_A'])       # (D, r)
    B = _get_data(self.params['lora_B'])       # (r, U)
    h = x_lora @ A                             # (N, r) 🔍 for grad_B
    self._cache_h = h
    lora_out = h @ B * self.scaling            # (N, U)
    base_out = x_flat @ W_dq                   # (N, U) — uses dequantized weight
    z_flat = base_out + lora_out
    if self.b_frozen is not None:
        z_flat = z_flat + self.b_frozen
    self._cache_z_pre = z_flat.reshape(*orig_shape[:-1], self.units)  # 🔍 for phi'
    z = z_flat.reshape(*orig_shape[:-1], self.units)
    if self.activation:
        out = self.activation(z)
        return Tensor(out) if isinstance(inputs, Tensor) else out
    return Tensor(z) if isinstance(inputs, Tensor) else z
```

🔍 **`W_dq`** (line ~185): Dequantized float copy of the base weight.
It is cached as `self._cache_W_dq` for `backward`'s `dx_W = gz @ W_dq^T`.
It is **not** stored permanently — only the 4-bit codes live in the layer.

🔍 **`b_frozen`** behaves identically to `LoRADense` — a frozen float bias
(if `use_bias=True`).

📐 **Memory walkthrough**: `W_dq` `(32,64)` float64 would be 16 KiB;
`W_codes` `(32,64)` uint8 (4-bit packed as uint8 here) is 2 KiB + absmax
overhead 128 B (double-quantized). ~8× compression.

### Step 5: `backward` — same as LoRA, but `dx` via `W_dq`

```python
def backward(self, grad_output):
    g_flat = ... # upstream, after activation chain if any
    self.grads['lora_B'] = self.scaling * (h.T @ g_flat)  # (r, U)
    dh = self.scaling * (g_flat @ B.T)                     # (N, r)
    self.grads['lora_A'] = x_lora.T @ dh                   # (D, r)
    dx_W = g_flat @ W_dq.T
    dx_lora = dh @ A.T
    dx = (dx_W + dx_lora).reshape(x.shape)
    return dx
```

🔍 **Line ~232–260**: The math is verbatim `LoRADense.backward`, except
`W_frozen` is replaced by `W_dq` (the dequantized weight). The absmax
quantization is *not* differentiated — the straight-through estimator is
not needed because the base weight is frozen.

### Step 6: `memory_bytes` — report compression

```python
def memory_bytes(self):
    n = self.W_codes.size                        # D*U
    base = n * 0.5                               # 4-bit
    num_blocks = self.W_absmax.size
    overhead = num_blocks * 1.0 + self.W_scales2.size * 4  # double-quant
    lora = A.size*8 + B.size*8                   # fp64
    return {'base_nf4': base, 'absmax_overhead': overhead, 'lora_fp64': lora}
```

🔍 **Line ~262–278**: Approximate byte counts. NF4 codes are counted as
0.5 bytes/param (4-bit). The example `qlora_finetune.py` calls this to
print a human-readable compression ratio.

### Step 7: `apply_qlora` — same interface as `apply_lora`

```python
def apply_qlora(model, rank=8, alpha=16, block_size=64,
                double_quant=True, target_modules=None, verbose=True):
    targets = [l for l in all_layers if isinstance(l, Dense)]
    for dense in targets:
        qlora = QLoRADense.from_dense(dense, rank=rank, ...,
                                       block_size=block_size,
                                       double_quant=double_quant)
        # replace in model.layers or via _replace_layer_q recursion
```

🔍 **`from_dense`** (line ~130): Quantizes the `Dense` weight with
`quantize_nf4` / `quantize_absmax` instead of copying it raw. Otherwise
identical to `LoRADense.from_dense`.

## Quantization utilities

File: `neutro/utils/quantization.py`

### `quantize_nf4` / `dequantize_nf4`

```python
def quantize_nf4(W, block_size=64):
    W_blocks = W.ravel().reshape(num_blocks, block_size)  # (B, bs)
    absmax = np.max(np.abs(W_blocks), axis=1)             # (B,)
    W_norm = W_blocks / absmax[:, None]                   # [-1, 1]
    diff = np.abs(W_norm[:, :, None] - NF4_CODEBOOK)      # (B, bs, 16)
    codes = np.argmin(diff, axis=2).astype(np.uint8)      # (B, bs)  0–15
```

🔍 **Line ~40–100**: For each block, divide by absmax to get `[-1,1]`,
then find the nearest NF4 codebook entry by brute-force distance to all
16 levels (vectorized). `dequantize_nf4` does the inverse: `NF4[code] * absmax`.

### `quantize_absmax` / `dequantize_absmax`

```python
def quantize_absmax(absmax, block_size=256):
    blocks = absmax.reshape(num_super, block_size)        # (S, sb)
    scales2 = np.max(np.abs(blocks), axis=1)              # (S,)
    normed = blocks / scales2[:, None]                    # [0, 1] (absmax ≥ 0)
    q = np.round(normed * 255).astype(np.uint8)           # [0, 255]
```

🔍 **Line ~115–170**: Uniform 8-bit quantization. Because absmax values
are non-negative, the range is `[0, s2]` rather than `[-s2, s2]`.
Dequant is `q / 255 * s2`.

### Codebook constant

```python
NF4_CODEBOOK = np.array([-1.0, -0.696, -0.525, ..., 0.723, 1.0])
```

🔍 **Line ~13–28**: The 16 NF4 levels from the QLoRA paper — quantiles of
`N(0,1)` mapped to `[-1, 1]`. These are optimal for weights that are
initially close to normal (Glorot/He initialization).

## Try it yourself

```python
from neutro.layers import QLoRADense, apply_qlora
from neutro.models import Sequential
from neutro.layers import Dense, Embedding, TransformerBlock, Softmax
import numpy as np

# Standalone
qlora = QLoRADense(32, rank=4, block_size=64, double_quant=True)
y = qlora(np.random.randn(2, 16))  # (2, 32)

# Model-level: same workflow as LoRA
model = Sequential([
    Embedding(100, 16, input_shape=(8,)),
    TransformerBlock(16, 2, 32),
    Dense(100), Softmax()
])
model(np.random.randint(0, 100, size=(2, 8)))  # build
model, adapted = apply_qlora(model, rank=8, block_size=64, double_quant=True)

# Memory report
for lyr in adapted:
    print(lyr.memory_bytes())

# Fine-tune, then dequant-merge for inspection
W_merged = adapted[0].get_merged_weight()  # dequant(W) + s*A@B
```

## What to read next

- **`neutro/layers/core/lora.md`** — LoRA (the non-quantized base).
- **`neutro/utils/quantization.py`** — full source for NF4 + double quant.
- **`neutro/layers/core/bitlinear.md`** — BitNet quantization (1-bit / 1.58-bit) for comparison.
- **`neutro/layers/core/dense.md`** — the `Dense` layer being quantized.
- **Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs"** (NeurIPS 2023).
