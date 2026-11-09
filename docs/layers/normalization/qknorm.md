# QKNorm — `neutro/layers/normalization/qknorm.py`

Per-head RMS norm for Q/K (Qwen3-style). `y = (x / rms(x)) * g`.
📐 Shape: `(..., D) → (..., D)`; typically `(B, S, H, Dh)`.

## Walkthrough

### `__init__/build`
`epsilon`; `gamma = Tensor(ones(D))` on last dim.
🔍 Why gain: lets heads re-scale after norm (like RMSNorm).

### `forward`
`rms = sqrt(mean(x²)+eps)` keepdims; `x/rms*gamma`.
🔍 Why tape default: pure Tensor ops, no manual backward.
