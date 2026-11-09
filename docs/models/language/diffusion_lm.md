# MaskedDiffusionLM — `neutro/models/language/diffusion_lm.py`

DiT-backed absorbing diffusion LM. Training: random `t→mask→logits`; loss
(`MaskedSparseCrossentropy`) averages only over masked tokens. Sampling:
iterative confidence-based unmasking.

## Walkthrough

### `__init__`
`Embedding + TimeMLP + Dense(dim) + DiTBlocks + RMSNorm + Dense(V)`; `scheduler` with `mask_id=V-1`.
🔍 Why `time_proj`: AdaLN needs cond dim `D`, TimeMLP outputs `time_dim`.

### `build`
Builds embeddings/time path on `(B,)`/`(B, time_dim)`; blocks on `[h_shape, c_shape]`; head on `h_shape`.

### `forward`
Train: `t~U(0,T)`, `(xt, mask)=scheduler.q_sample`, cache `_last_mask`; eval: `t=0`, raw ids.
`h=Emb(xt)`; `cond=time_proj(time_mlp(t))`; loop `DiT`; `head(norm(h))`.
📐 `(B, S)→(B, S, D)→(B, S, V)`.

### `backward`
Head→norm→ reversed DiT (each returns `[dh, dcond]` list via tape) → embedding.
🔍 Why list check: `DiTBlock` takes `[h, cond]`, so tape grad is a 2-list.

### `sample`
Prefix-fixed + per-step `per_step = remaining//steps` lowest-confidence unmasking with temperature.
