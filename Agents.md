# Neutro Agent Guidelines

You are an agent working on `neutro`, an "intentionally naive" and educational implementation of Keras-like APIs using only NumPy and SciPy.

## Core Principles

1.  **NumPy Only**: Avoid using specialized deep learning frameworks (TensorFlow, PyTorch, JAX). All logic must be implemented in pure NumPy.
2.  **Keras API Fidelity**: Maintain strict compatibility with Keras/TensorFlow APIs (`compile`, `fit`, `predict`, `evaluate`, `summary`, `Sequential`, `Model`).
3.  **Educational Clarity**: Code should be readable and reflect the underlying mathematical algorithms (e.g., FlashAttention, MoE routing, RoPE). Use clear variable names and minimal but impactful comments.
4.  **No Magic**: Avoid complex meta-programming or obscure libraries. If a layer needs a backward pass, implement it explicitly.
5.  **Nested Training**: Ensure that nested layers (layers within blocks) are discovered and updated by the optimizer. Use `Layer.sublayers` to traverse the hierarchy.

## Implementation Details

### Layers
- All layers inherit from `neutro.layers.base.Layer`.
- Must implement `forward(inputs, training=False)` and `backward(grad_output)`.
- Must implement `build(input_shape)` to initialize parameters in `self.params`.
- Should implement `compute_output_shape(input_shape)` for model summaries.

### Models
- Use `neutro.models.base_model.Sequential` or `Model`.
- `Model.forward` handles the sequence of layer calls.
- `Model.generate` supports autoregressive inference with KV Caching.

### Attention & Transformers
- `FlashAttention` is the preferred attention mechanism.
- `KVCache` is used for efficient generation.
- `MLA` (Multi-Head Latent Attention) should be used for DeepSeek-style models.

## Testing
- Aim for >90% test coverage.
- Use `pytest`.
- Compare output/gradients against small numerical finite differences or known outputs when possible.
