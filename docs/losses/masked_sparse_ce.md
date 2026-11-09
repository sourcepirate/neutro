# MaskedSparseCrossentropy — `neutro/losses/masked_sparse_ce.py`

Averages NLL only over `model._last_mask`. `gradient` masks one-hot grad and divides by `mask.sum`.
🔍 Why model ref: mask is sampled inside `forward`, not passed as `y`; loss reads the cache.
📐 `y (B, S)`, `logits (B, S, V) → grad (B, S, V)`.
