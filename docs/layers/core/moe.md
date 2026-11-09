# MoELayer — `neutro/layers/core/moe.py`

Top-k Mixture of Experts, dense evaluation for clarity. Replaces the old
numpy-manual version with autograd `Tensor` ops.

Math: `logits = X @ W_r`; `P = softmax(logits)`; top-k mask `M`;
`w = (P*M)/sum(P*M)`; `y = Σ_e w_e * Expert_e(X)`.
📐 Shapes: `X (T, D) @ (D, E) → (T, E)`; expert relu path `(T, D)→(T, H)→(T, D)`; SwiGLU path same via `SwiGLUMLP`.

## Walkthrough (`moe.py`)

### `__init__`
`num_experts/top_k/expert_type ∈ {relu, swiglu}/expert_hidden_dim/capacity_factor/aux_loss_coef`.
🔍 Why `expert_type`: Qwen/DeepSeek need SwiGLU experts; vision toy can use relu.

### `build`
`router_weight = Tensor(D, E)`; per expert builds `Dense(H)+Dense(D)` or `SwiGLUMLP(D, H)`.
🔍 Why `Tensor`: router participates in `GradientTape`; old `np.random` version could not.

### `forward`
Flattens `(B, S, D)→(T, D)`; `probs = softmax(X@W)`; numpy top-k mask (non-differentiable indices, differentiable weights); renormalize; loop experts, accumulate `w_e * Expert_e(X)` with Tensor ops; reshape back.
🔍 Why dense loop not scatter: avoids in-place scatter grads; every op stays on tape.
🔍 Why store `_last_frac/_last_mean_prob/aux_loss_value`: router utilization report + `aux_loss` without extra forward.

### `aux_loss`
`num_experts * Σ(frac_e * mean_prob_e)`, scaled by `aux_loss_coef` by caller.
