# Diffusion layers — `neutro/layers/diffusion/`

## TimeMLPEmbedding (`time_mlp.py`)
Sinusoid `t→(B, D)` then `Dense(H)→SiLU→Dense(D)`.
📐 `(B,) → (B, D) → (B, H) → (B, D)`.
🔍 Why MLP after sinusoid: raw sinusoids are fixed; MLP learns schedule conditioning. `build` shapes `fc1/fc2` from `dim`.

## AdaLN / AdaLNZero (`adaln.py/33`)
`y = LayerNorm(x)*(1+scale(c)) + shift(c)`; Zero adds `gate(c)` initialized to 0.
`build([x_shape, c_shape])` sets `Dense(D)` units then zero-fills `W/b`.
🔍 Why zero-init: block starts as identity, critical for DiT training stability.
📐 `x (B, S, D)`, `c (B, C) → (B, 1, D)` broadcast.

## DiTBlock (`dit_block.py`)
`h = x + gate1*Attn(AdaLN(x,c))`; `out = h + gate2*MLP(AdaLN(h,c))`; attention is bidirectional `FlashAttention`.
🔍 Why `gate = gate_proj(c)` reshaped `(B,1,D)`: per-token modulation from one cond vector.

## DiscreteDiffusionScheduler (`discrete_scheduler.py`)
Absorbing-mask: `P(mask|t) = 1 - alpha_bar(t)`, linear or cosine `alpha_bar`.
`q_sample(x0, t)` returns `(xt, mask)` with `xt[mask] = mask_id`.
🔍 Why store `alpha_bar` array: `mask_prob(t)` is O(1) lookup; schedule matches LLaDA/MDLM.
