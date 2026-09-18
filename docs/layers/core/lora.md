# LoRADense Layer

## What does this layer do?

LoRA (Low-Rank Adaptation, Hu et al. 2022) lets you fine-tune a large
pre-trained model by updating only a *tiny* fraction of its parameters.

Instead of updating the full weight matrix `W` (shape `D × U`), LoRA
**freezes** `W` and learns a low-rank residual:

```
W' = W_frozen + (alpha / r) * A @ B
```

- `A` has shape `(D, r)` — the "down-projection".
- `B` has shape `(r, U)` — the "up-projection".
- `r` (rank) is the bottleneck, typically 4–64.  When `r << min(D,U)`,
  `A` and `B` together have far fewer parameters than `W`.
- `alpha` is a scaling hyperparameter; the effective scale is `s = alpha / r`.
- `B` is zero-initialized so that `A @ B = 0` at the start — the adapted
  model behaves *identically* to the base model before any training.

During forward:

```
y = activation( x @ W_frozen + b_frozen + s * (x @ A) @ B )
```

Only `A` and `B` receive gradients; `W_frozen` and `b_frozen` stay fixed.
After fine-tuning you can **merge** the adapter (`W_frozen ← W_frozen + s·A@B`)
and run a single matmul at inference.

Think of it as writing a correction in the margins of a textbook — the
original text stays untouched, and the correction is small, cheap, and
reversible.

## The math, in plain English

```
Let x : (B, D)        — batch of B inputs, each D-dimensional
    W : (D, U)        — frozen base weight
    b : (U,)          — frozen bias (optional)
    A : (D, r)        — trainable down-projection
    B : (r, U)        — trainable up-projection
    s : scalar        — alpha / r

    h        = x @ A              (B, D) @ (D, r) → (B, r)   — compress
    lora     = h @ B * s          (B, r) @ (r, U) → (B, U)   — expand + scale
    z        = x @ W + b + lora   (B, U)                      — base + adapter
    y        = phi(z)             element-wise activation
```

After flattening time/sequence dims (`(B, T, D) → (B·T, D)`) the same
formulas apply.

### How gradients flow backward

```
Let g = dL/dy                          — upstream gradient (B, U)
    gz = g ⊙ phi'(z)  (if activation) — dL/dz (B, U), chain rule

For the LoRA parameters:
    dL/dB = s * h^T @ gz               (r, B) @ (B, U) → (r, U)
    dL/dh = s * gz @ B^T               (B, U) @ (U, r) → (B, r)
    dL/dA = x_lora^T @ dL/dh           (D, B) @ (B, r) → (D, r)

For the input (passed to previous layer):
    dL/dx_W    = gz @ W^T              (B, U) @ (U, D) → (B, D)  — via frozen W
    dL/dx_lora = dL/dh @ A^T           (B, r) @ (r, D) → (B, D)  — via LoRA
    dL/dx      = dL/dx_W + dL/dx_lora
    (if dropout mask M was applied: dL/dx_lora *= M)
```

Frozen `W` and `b` receive **no** gradient — they are plain `numpy`
arrays, not in `self.params`, so the optimizer never sees them.

## Walking through the code

File: `neutro/layers/core/lora.py`

### Step 1: `__init__` — choosing the adapter size

```python
class LoRADense(Layer):
    def __init__(self, units, rank=8, alpha=16, dropout=0.0,
                 use_bias=True, activation=None,
                 kernel_initializer='glorot_uniform', ...):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.dropout_rate = float(dropout)
        ...
        self.W_frozen = None  # set in build()
        self.b_frozen = None
```

🔍 **Line 50–62**: No weights are allocated here. `units` and `rank`
are hyperparameters we choose; `input_dim` is unknown until `build`.
`self.scaling = alpha / rank` is the constant `s` from the math above.

🔍 **Line 78–80**: `W_frozen` / `b_frozen` will be plain `numpy` arrays,
**not** `Tensor` objects and **not** in `self.params`. This is the
mechanism that freezes them — the optimizer iterates `self.params` only.

### Step 2: `build` — allocating frozen + trainable weights

```python
def build(self, input_shape):
    self.input_dim = input_shape[-1]
    self.W_frozen = self.kernel_initializer((self.input_dim, self.units)).astype(float)
    if self.use_bias:
        self.b_frozen = self.bias_initializer((self.units,)).astype(float)
    self.params['lora_A'] = Tensor(self.lora_initializer((self.input_dim, self.rank)))
    self.params['lora_B'] = Tensor(np.zeros((self.rank, self.units)))
    super().build(input_shape)
```

🔍 **Line 87**: `input_shape[-1]` is `D`. For a 3-D sequence
`(B, T, D)`, we still get `D` — Dense/LoRA operates on the last axis.

🔍 **Line 90**: `W_frozen` is `numpy`, shape `(D, U)`. Uses the same
initializers as `Dense` (Glorot uniform by default).

🔍 **Line 94–98**: `lora_A` `(D, r)` — Glorot/Kaiming init so the adapter
starts with a sensible scale. `lora_B` `(r, U)` — **zeros** so the
adapter contributes nothing at initialization (delta = 0). This is why
`apply_lora` copies weights and the model output is unchanged right after
adaptation.

📐 **Shape walkthrough**: For `input_dim=32, units=64, rank=8`:
`W_frozen` `(32, 64)` = 2048 frozen params;
`A` `(32, 8)` = 256 + `B` `(8, 64)` = 512 → 768 trainable params
(37.5% of the base).

### Step 3: `from_dense` — wrapping an existing Dense

```python
@classmethod
def from_dense(cls, dense_layer, rank=8, alpha=16, ...):
    obj = cls(units=dense_layer.units, rank=rank, alpha=alpha, ...)
    obj.W_frozen = _get_data(dense_layer.params['W']).copy()
    if use_bias and 'b' in dense_layer.params:
        obj.b_frozen = _get_data(dense_layer.params['b']).copy()
    obj.params['lora_A'] = Tensor(obj.lora_initializer((obj.input_dim, obj.rank)))
    obj.params['lora_B'] = Tensor(np.zeros((obj.rank, obj.units)))
    return obj
```

🔍 **Line 114–158**: `from_dense` is how `apply_lora` injects LoRA into a
built model. It copies the *data* out of the old `Dense` weight
(`_get_data` unwraps `Tensor → ndarray`) so the new layer's frozen weight
is a snapshot, not a reference.

### Step 4: `forward` — base + LoRA branch

```python
def forward(self, inputs, training=False):
    x = inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs)
    self._cache_x = x                          # 🔍 for backward: x and dx
    x_flat = x.reshape(-1, self.input_dim)     # (N, D)
    if training and self.dropout_rate > 0:
        mask = (np.random.rand(*x_flat.shape) >= self.dropout_rate)
        mask /= (1 - self.dropout_rate)
        self._cache_dropout_mask = mask        # 🔍 scale grad through dropout
        x_lora = x_flat * mask
    else:
        x_lora = x_flat
    A = _get_data(self.params['lora_A'])       # (D, r)
    B = _get_data(self.params['lora_B'])       # (r, U)
    h = x_lora @ A                             # (N, r)  🔍 cached for grad_B
    self._cache_h = h
    lora_out = h @ B * self.scaling            # (N, U)
    base_out = x_flat @ self.W_frozen          # (N, U)
    z_flat = base_out + lora_out
    if self.b_frozen is not None:
        z_flat = z_flat + self.b_frozen
    self._cache_z_pre = z_flat.reshape(*orig_shape[:-1], self.units)  # 🔍 for activation grad
    z = z_flat.reshape(*orig_shape[:-1], self.units)
    if self.activation:
        out = self.activation(z)               # e.g. ReLU
        return Tensor(out) if isinstance(inputs, Tensor) else out
    return Tensor(z) if isinstance(inputs, Tensor) else z
```

🔍 **`self._cache_x`** (line 200): Full input `(B, ..., D)` — needed to
compute `dL/dA = x^T @ dL/dh` and to reshape `dx` back to the original
rank. Also needed for `dx = gz @ W^T`.

🔍 **`self._cache_dropout_mask`** (line 213): Inverted dropout mask.
During `backward`, `dx_lora *= mask` re-applies the same stochastic
scaling so gradients match the forward that was actually computed.

🔍 **`self._cache_h`** (line 226): The bottleneck `h = x_lora @ A`
`(N, r)`. Needed for `dL/dB = s * h^T @ gz`. Recomputing it would be
wasteful.

🔍 **`self._cache_z_pre`** (line 237): Pre-activation `z = base + lora`
`(..., U)` before the activation. For `backward`, `phi'(z)` needs this
value (e.g. `ReLU.gradient(z) = (z > 0)`).

📐 **Shape walkthrough (flattened)**: `x_flat` `(N=64, D=32)` →
`h = (64,32) @ (32,8) → (64,8)` → `lora = (64,8) @ (8,64) → (64,64)` →
`base = (64,32) @ (32,64) → (64,64)` → `z = (64,64)` → reshape to
`(B, T, 64)`.

### Step 5: `backward` — only A and B learn

```python
def backward(self, grad_output):
    g = _to_numpy(grad_output)          # (..., U)
    g_flat = g.reshape(-1, self.units) # (N, U)
    if self.activation is not None:
        z_flat = self._cache_z_pre.reshape(-1, self.units)
        if hasattr(self.activation, 'gradient_fast'):
            g_flat = self.activation.gradient_fast(z_flat, g_flat)
        else:
            g_flat = g_flat * self.activation.gradient(z_flat)  # ⊙ phi'(z)

    h = self._cache_h           # (N, r)
    self.grads['lora_B'] = self.scaling * (h.T @ g_flat)   # (r, U)
    dh = self.scaling * (g_flat @ B.T)                     # (N, r)
    self.grads['lora_A'] = x_lora.T @ dh                   # (D, r)

    dx_W = g_flat @ self.W_frozen.T   # (N, D)
    dx_lora = dh @ A.T                # (N, D)
    if self._cache_dropout_mask is not None:
        dx_lora = dx_lora * self._cache_dropout_mask
    dx = (dx_W + dx_lora).reshape(x.shape)
    return dx
```

🔍 **`self.grads['lora_B']`** (line 296): Implements `dL/dB = s·h^T·gz`.
Shape `(r, U)` — same as `B`. No grad for `W_frozen`.

🔍 **`dh` and `grads['lora_A']`** (line 299–303): Chain through `B`:
`dL/dh = s·gz·B^T`, then `dL/dA = x_lora^T·dL/dh`.

📐 **Gradient shapes**: `h^T·gz` is `(r,N)@(N,U)→(r,U)` for `B`;
`x_lora^T·dh` is `(D,N)@(N,r)→(D,r)` for `A`.

🔍 **Input gradient** (line 307–316): Sum of two paths — through frozen
`W` and through the adapter. Both contribute to the gradient that flows
to the previous layer.

### Step 6: `merge_weights` / `get_merged_weight`

```python
def merge_weights(self):
    delta = self.scaling * (A @ B)      # (D, U)
    self.W_frozen = self.W_frozen + delta
    self.params['lora_B'].data[:] = 0.0  # so second forward == merged
```

🔍 **Line 164–176**: Folding the adapter into the base weight makes
inference a single matmul. Because `B` is zeroed, subsequent
`h@B` is zero, so `x@W_new + 0` equals the old `x@W + s·xAB`.

### Step 7: `apply_lora` — injecting LoRA into a built model

```python
def apply_lora(model, rank=8, alpha=16, target_modules=None, verbose=True):
    all_layers = model._get_all_layers()
    targets = [l for l in all_layers if isinstance(l, Dense)]  # auto-detect
    for dense in targets:
        lora = LoRADense.from_dense(dense, rank=rank, alpha=alpha)
        # replace in parent: model.layers list or via _replace_layer recursion
```

🔍 **`target_modules=None`** → adapt every `Dense`. For transformer
models this covers the FFN (`TransformerBlock.ffn`) and the output
projection. Pass `target_modules=["output"]` to adapt only layers whose
`name` contains `"output"`. Or pass explicit layer objects.

🔍 **`_replace_layer`** handles `Sequential.layers`, `TransformerBlock.ffn`
lists, and arbitrary nested dicts.

## Putting it all together

1. Build or load a pre-trained model and call it once to trigger `build`.
2. `apply_lora(model, rank=8, alpha=16, verbose=True)` snapshots each
   `Dense` weight into a new `LoRADense.W_frozen` and inserts the adapter.
3. `model.compile` with a fresh optimizer — it will only see `lora_A/B`.
4. `model.fit` fine-tunes; `backward` updates `A, B` only.
5. `layer.merge_weights()` folds `s·A@B` into `W_frozen` for deployment;
   `layer.get_merged_weight()` peeks without mutating.

## Try it yourself

```python
from neutro.models import Sequential
from neutro.layers import Dense
from neutro.layers import LoRADense, apply_lora
import numpy as np

# Pre-trained model
model = Sequential([Dense(32, activation='relu', input_shape=(16,)), Dense(4)])
x = np.random.randn(8, 16)
model(x)  # build

# Inject LoRA (rank=4 — only 80 trainable params vs 544 frozen)
model, adapted = apply_lora(model, rank=4, alpha=8, verbose=True)

# Forward is unchanged at init (B=0)
out = model(x)

# After training, merge for inference
for lyr in adapted:
    lyr.merge_weights()
out_merged = model(x)
assert np.allclose(out, out_merged, atol=1e-6)

# Standalone usage
lora = LoRADense(10, rank=4, alpha=8, activation='relu')
y = lora(np.random.randn(5, 8))
print(y.shape)  # (5, 10)
```

## What to read next

- **`neutro/utils/quantization.md`** (if present) — NF4 quantization used by QLoRA.
- **`neutro/layers/core/qlora.md`** — QLoRA: frozen base in 4-bit NF4 + LoRA.
- **`neutro/layers/core/dense.md`** — the `Dense` layer that LoRA wraps.
- **`neutro/layers/core/bitlinear.md`** — another quantized-linear variant.
- **Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"** (ICLR 2022).
