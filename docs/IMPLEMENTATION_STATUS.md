# Implementation Status

Neutro is an educational library. Some components are full implementations,
some are deliberately simplified, and some keep an API-compatible name while
using a simpler algorithm. This page says which is which, so you are never
surprised by the code.

**Legend**

- **Implemented** — the algorithm is here, in NumPy.
- **Simplified** — the shape/API is real, but a faster or more complex part is
  replaced with the straightforward version.
- **Alias** — the name resolves to another implementation; it is not a distinct
  architecture.
- **Experimental** — works, but the interface may change.

---

## Attention

| Component | Status | Notes |
|-----------|--------|-------|
| `MultiHeadAttention` | Implemented | Scaled dot-product attention, optional KV cache. |
| `MultiQueryAttention` | Implemented | One KV head shared by all query heads. |
| `GroupedQueryAttention` | Implemented | KV heads shared within groups. |
| `MultiHeadLatentAttention` | Implemented | Low-rank Q/KV compression, latent KV cache. |
| `FlashAttention` | **Simplified** | Uses the same projections and output shape, but computes vanilla `QK^T -> softmax -> V`. The tiled/online-softmax algorithm is **not** implemented. See the docstring in `neutro/layers/attention/flash_attention.py`. |
| `PagedAttention` | Experimental | Educational paged KV cache; not a production serving kernel. |
| `KVCache` | Implemented | Dynamic grow-on-append cache for inference. |

## Language models

| Component | Status | Notes |
|-----------|--------|-------|
| `GPT1`, `GPT2` | Simplified | Small educational stacks; not full GPT checkpoints. |
| `LlamaTiny` | Simplified | RMSNorm + RoPE attention + SwiGLU, tiny dimensions. |
| `Qwen1Tiny`, `Qwen2Tiny`, `QwenTiny` | Simplified | Qwen-style blocks with GQA. |
| `Qwen3Tiny` | **Alias** | `Qwen2Tiny` with two extra layers. Not the real Qwen3 architecture. |
| `Qwen3MoETiny` | Experimental | Qwen-style block with MoE FFN and shared expert. |
| `DeepSeekV1Tiny`, `DeepSeekV2Tiny` | Simplified | MoE blocks; V2 adds MLA. |
| `DeepSeekV3Tiny` | Simplified | More experts/top-k than V2. |
| `DeepSeekV4Tiny` | **Alias** | `DeepSeekV3Tiny` with one extra layer. Not a released architecture. |
| `BitNetTiny` | Experimental | BitLinear weights; see the BitNet block. |
| `MaskedDiffusionLM` | Experimental | Masked diffusion LM with a DiT backbone. |

## Vision models

| Component | Status | Notes |
|-----------|--------|-------|
| `AlexNet`, `VGG16`, `VGG19` | Simplified | Architecture-shaped; not pretrained. |
| `VAE` | Simplified | Convolutional encoder + Dense decoder. |
| `UNet` | Simplified | Time-conditioned UNet for diffusion demos. |
| `DiffusionModel` | Simplified | Wraps a UNet with a Gaussian diffusion process. |

## Layers and building blocks

| Component | Status | Notes |
|-----------|--------|-------|
| `Dense`, `Conv1D`, `Conv2D`, `MaxPooling2D` | Implemented | Tape-backed gradients; `im2col` convolutions. |
| `Dropout` | Implemented | Inverted dropout via a custom `Function`. |
| `LayerNormalization`, `RMSNorm`, `BatchNormalization`, `GroupNormalization` | Implemented | Tape-backed. |
| `LSTM`, `GRU`, `SimpleRNN` | Implemented | Loop over timesteps. |
| `MoELayer` | Simplified | Dense evaluation: every token is scored by the router; only experts with assigned tokens run. No capacity dropping. |
| `SwiGLUMLP` | Implemented | Llama-style gated MLP. |
| `LoRADense`, `QLoRADense` | Experimental | Manual backward; QLoRA stores NF4 base weights. |
| `BitLinear` | Experimental | Manual backward with quantized forward. |
| `Reparameterization` | Implemented | VAE reparameterization trick. |
| `AdaLN`, `DiTBlock`, `TimeMLPEmbedding` | Experimental | Diffusion transformer building blocks. |

## Backward strategies

| Strategy | Layers | Why |
|----------|--------|-----|
| **Tape-backed fallback** (default) | `Dense`, `Conv*`, attention, normalization, recurrent, transformer, most others | Write `forward` once; `Layer.backward` replays it under `GradientTape`. Best for learning the forward computation. |
| **Manual `backward`** | `BitLinear`, `LoRADense`, `QLoRADense`, `UNet`, `VAE`, `MaskedDiffusionLM` | Explicit chain-rule code where the gradient mechanics are the lesson, or where feed-forward replay is awkward. |

The beginner path is tape-first. Manual backprop is an advanced topic; see
[tutorial 02](tutorials/02-gradients-and-the-tape.md) for the tape story and
the layer docstrings for the manual variants.

---

If a page or docstring claims a feature is more complete than this table says,
treat this table as the source of truth and open an issue.
