# Tutorial 07 — LoRA Fine-Tuning

**Time:** about 20 minutes. **Prerequisites:** [Tutorial 01](01-first-model.md) and [Tutorial 05](05-manual-backprop.md).

LoRA (**Low-Rank Adaptation**) fine-tunes a model by freezing its base weights
and learning a small low-rank update. It is the standard way to adapt a large
model without training every parameter.

Run the companion script:

```bash
python examples/07_lora_tour.py
```

Expected output:

```text
Parameter counts
  Dense params:         2080
  LoRA trainable params: 384
  reduction:            81.5%
  trainable keys:       ['lora_A', 'lora_B']

Initial output matches the frozen base:
  True

After merge_weights()
  output unchanged:     True
  merged W shape:       (64, 32)
  lora_B zeroed:        True

Manual backward
  grad shapes:          {'lora_B': (4, 32), 'lora_A': (64, 4)}
  input grad shape:     (2, 64)
```

---

## 1. The idea

A Dense layer has a base weight `W` of shape `(D, U)`. LoRA freezes `W` and adds
a low-rank update:

```text
W' = W_frozen + (alpha / rank) * A @ B

A: (D, rank)
B: (rank, U)
```

The forward pass becomes:

```text
y = x @ W_frozen + b + s * (x @ A) @ B
s = alpha / rank
```

`rank` is much smaller than `D` and `U`, so `A` and `B` together have far fewer
parameters than `W`.

`B` is initialized to **zero**, so the adapter starts as a no-op:

```text
x @ A @ B = 0  at initialization
```

That is why the LoRA output initially matches the frozen base exactly.

---

## 2. Trainable vs frozen

`LoRADense` stores:

- `W_frozen`, `b_frozen` — plain arrays, **not** in `params`.
- `lora_A`, `lora_B` — `Tensor` objects in `params`.

The optimizer only walks `layer.params`, so it updates `A` and `B` and leaves
the base weights alone.

```python
print(sorted(lora.params))
# ['lora_A', 'lora_B']
```

Parameter count:

```text
base Dense:  D*U + U          = 64*32 + 32 = 2080
LoRA:        D*r + r*U        = 64*4 + 4*32 = 384
reduction:   1 - 384/2080     = 81.5%
```

The saving grows as `D` and `U` grow, because the adapter is `O((D+U)r)` while
the base is `O(DU)`.

---

## 3. Build and compare

```python
from neutro.layers import Dense, LoRADense

base = Dense(units=32)
base.build((2, 64))

lora = LoRADense(units=32, rank=4, alpha=8)
lora.build((2, 64))

# Copy the base weights into the frozen slots.
lora.W_frozen = base.params["W"].data.copy()
lora.b_frozen = base.params["b"].data.copy()
```

At initialization:

```python
print(np.allclose(base(x).data, lora(x).data))   # True
```

Why? `B` is zero, so the LoRA branch contributes nothing.

---

## 4. Gradients

`LoRADense` writes its own `backward`:

```text
dL/dB = s * h.T @ g
dL/dh = s * g @ B.T
dL/dA = x_lora.T @ dL/dh
dL/dx = g @ W_frozen.T + dL/dh @ A.T
```

where `h = x @ A` and `g = dL/dy`.

```python
lora.backward(upstream)

print(lora.grads["lora_A"].shape)   # (64, 4)
print(lora.grads["lora_B"].shape)   # (4, 32)
```

The base weight has no gradient because it is frozen.

---

## 5. Merge the adapter

After fine-tuning, you can fold the adapter into the base weight:

```text
W_frozen <- W_frozen + s * A @ B
B <- 0
```

Then forward is a single matmul again, with no LoRA branch:

```python
before = lora(x).data.copy()
lora.merge_weights()
after = lora(x).data
print(np.allclose(before, after))   # True
```

`merge_weights()` mutates `W_frozen` and zeroes `B`. `get_merged_weight()`
returns the merged weight without mutating state.

This is useful for deployment: you pay the LoRA cost during training, then
serve the merged base layer.

---

## 6. Where it fits in a model

Typical workflow:

1. Train or load a base model.
2. Replace target `Dense` layers with `LoRADense` using `apply_lora`.
3. Fine-tune only `lora_A` / `lora_B`.
4. Evaluate.
5. Merge for inference.

See `examples/lora_finetune.py` for a full pre-train → adapt → merge → generate
script, and `QLoRADense` for the 4-bit NF4 variant.

---

## 7. Your turn

1. Change `rank` from 4 to 1. How many trainable parameters remain?
2. Change `alpha`. Does it change the parameter count? What does it change?
3. Remove the zero initialization of `B`. Does the initial output still match
   the base?
4. After `merge_weights()`, can you continue training the same adapter? What
   would you need to store?
5. Compare `LoRADense` with a full `Dense` on a small regression task. Which
   reaches a lower training loss? Which uses fewer parameters?

<details>
<summary>Answers</summary>

1. `64*1 + 1*32 = 96` trainable parameters.
2. No. `alpha` scales the adapter's contribution (`alpha / rank`); it does not
   add parameters.
3. No. A non-zero `B` changes the output immediately.
4. `merge_weights()` zeroes `B` and folds the delta into `W_frozen`. To keep
   training, you would need to store the original delta or avoid merging until
   inference.
5. A full Dense usually fits a small task more easily because it has more
   parameters. LoRA trades some capacity for a much smaller trainable set.

</details>

---

## What to read next

- [LoRA layer walkthrough](../layers/core/lora.md)
- [QLoRA layer walkthrough](../layers/core/qlora.md)
- [Examples: LoRA fine-tuning](https://github.com/sourcepirate/neutro/blob/main/examples/lora_finetune.py)
- [Examples: QLoRA fine-tuning](https://github.com/sourcepirate/neutro/blob/main/examples/qlora_finetune.py)
- [Implementation status](../IMPLEMENTATION_STATUS.md)
