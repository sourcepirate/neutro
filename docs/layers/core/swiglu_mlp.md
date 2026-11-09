# SwiGLUMLP — `neutro/layers/core/swiglu_mlp.py`

Generic SwiGLU feed-forward used by Llama, Qwen, and MoE experts.

Math: `y = (SiLU(x @ W1) * (x @ W3)) @ W2`.
📐 Shapes: `(B, S, D) @ (D, H) → (B, S, H)`; gate*value `(B, S, H)`; `@ (H, D) → (B, S, D)`.

## Walkthrough (`swiglu_mlp.py`)

### `__init__`
Stores `dim`/`hidden_dim`, creates `w1/w2/w3 = Dense`, `silu = SiLU`.
🔍 Why store sublayers as attributes: `Layer.sublayers` traverses them so the optimizer finds `W1/W2/W3`.

### `build`
Builds `w1` on `input_shape`, `w2` on `(..., H)`, `w3` on `input_shape`.
🔍 Why 3D-safe shape: supports `(B, S, D)` and `(B, D)` via `input_shape[:-1]`.

### `forward`
`gate = w1(x)`; `activated = silu(gate)`; `value = w3(x)`; `return w2(activated * value)`.
🔍 Why no manual `backward`: all ops are autograd `Tensor`s; `Layer.backward` re-runs this inside `GradientTape`.
