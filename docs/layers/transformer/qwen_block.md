# QwenBlock — `neutro/layers/transformer/qwen_block.py`

Qwen3-style decoder: `h = x + Attn(RMSNorm(x))`, `out = h + SwiGLU(RMSNorm(h))`,
with per-head QK-Norm reshape before attention and RoPE inside `FlashAttention`.
📐 Shapes: `(B, S, D)` throughout; QKNorm view `(B, S, H, Dh)`.

## Walkthrough

### `__init__`
`FlashAttention(H, D, use_rope=True)`, `RMSNorm x2`, `SwiGLUMLP(D, 3.5D)`, optional `QKNorm x2`.
🔍 Why sublayer attrs: discovered via `sublayers` for optimizer step.

### `build`
Builds attention/norms/MLP; QKNorm built on `(B, S, H, Dh)`.

### `forward`
Norm → optional head-reshape + `q_norm` → attention (+KV-cache passthrough) → residual → MLP residual.
🔍 Why reshape QKNorm: `FlashAttention` owns Q/K projections, so we norm the head-view hidden state as an educational proxy.
