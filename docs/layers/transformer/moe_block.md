# MoETransformerBlock — `neutro/layers/transformer/moe_block.py`

Generic `h = x + Attn(Norm(x))`; `out = h + MoE(Norm(h)) + Shared(Dense(Norm(h)))`.
Backs `DeepSeek` and `Qwen3MoE`. Set `expert_type='swiglu'`, `use_mla=True` for DeepSeek-V2+.

## Walkthrough

### `__init__`
Picks `FlashAttention` or `MultiHeadLatentAttention`; `MoELayer`; optional shared `Dense(D)`.
### `build`
Builds all five in `input_shape`.
### `forward`
Try attention with `(mask, kv_cache, layer_id)`, fallback to `(x, training)` for MLA compat.
🔍 Why `aux_loss` passthrough: exposes `moe.aux_loss` for loss balancing.
📐 Shapes: `(B, S, D)` in/out.
