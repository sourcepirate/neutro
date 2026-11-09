# API and Naming Conventions

Neutro is a teaching codebase, so consistency matters more than cleverness. This
page records the conventions used across layers and models. Follow them when
adding new code; if you must break one, document why.

---

## Parameter names

Parameters live in `layer.params` and are usually `Tensor` objects.

| Name | Shape | Used by |
|------|-------|---------|
| `W` | `(input_dim, units)` | `Dense`, `Conv1D`, `Conv2D` kernels, `BitLinear` |
| `b` | `(units,)` | Dense/conv biases |
| `gamma` | `(features,)` or broadcastable | LayerNorm, RMSNorm, BatchNorm, GroupNorm, QKNorm |
| `beta` | `(features,)` or broadcastable | LayerNorm, BatchNorm, GroupNorm |
| `embeddings` | `(vocab, dim)` | `Embedding` |
| `Wq`, `Wk`, `Wv`, `Wo` | projection matrices | attention |
| `router_weight` | `(dim, num_experts)` | `MoELayer` |
| `lora_A`, `lora_B` | `(D, r)`, `(r, U)` | `LoRADense`, `QLoRADense` |

Rules:

- Prefer short, conventional names (`W`, `b`, `gamma`, `beta`) over
  `weight_matrix` or `bias_vector`.
- If a layer needs a second matrix, name it by role (`Wq`, `Wk`, `Wv`, `Wo`),
  not by index.
- Frozen parameters should not sit in `params`; the optimizer updates
  everything in `params`. `LoRADense` keeps `W_frozen` as a separate attribute
  on purpose.

---

## Attribute names

| Concept | Preferred attribute | Notes |
|---------|---------------------|-------|
| Attention sublayer | `self.attention` | `TransformerBlock` keeps `self.att` as a backward-compatible alias. |
| Feed-forward stack | `self.ffn` | Usually a list of layers. |
| Norms | `self.layernorm1`, `self.layernorm2` or `self.norm1`, `self.norm2` | Match the forward order. |
| Dropout | `self.dropout1`, `self.dropout2` | Match the sublayer they follow. |
| Context/conditioning | `self.cond` | Inputs to adaptive norm layers. |
| Cached input (manual backward only) | `self.inputs` | Only needed when `backward` is hand-written. |

`Layer.sublayers` discovers layers stored in attributes, lists, tuples, sets,
and dicts. Prefer normal attributes and lists over custom registration.

---

## Constructor arguments

| Argument | Meaning |
|----------|---------|
| `units` | Dense output width `U`. |
| `dim`, `embed_dim` | Model/embedding width `D`. |
| `num_heads` | Attention heads `H`. |
| `head_dim` | Per-head width `d`, often `D / H`. |
| `key_dim` | Total projected width in attention; `key_dim = H * d`. |
| `ff_dim`, `hidden_dim` | Feed-forward expansion width. |
| `rate`, `dropout` | Dropout probability. |
| `epsilon` | Numerical-stability constant in normalization. |
| `data_format` | `"channels_last"` or `"channels_first"` for image layers. |
| `input_shape` | Shape without the batch axis; the first call can also infer it. |

Factory functions for tiny language/vision models use the `Tiny` suffix
(`GPT2Tiny`, `LlamaTiny`, `Qwen2Tiny`, `DeepSeekV2Tiny`). If a name is an alias
for another factory, say so in the docstring and in
[Implementation status](IMPLEMENTATION_STATUS.md). `GPT1`/`GPT2` keep their
historical names, with `GPT1Tiny`/`GPT2Tiny` as explicit aliases.

---

## Shape notation

| Symbol | Meaning |
|--------|---------|
| `B` | batch size |
| `S`, `T` | sequence length / timesteps |
| `D` | model/embedding width |
| `H` | attention heads |
| `d` | per-head width |
| `V` | vocabulary size |
| `C` | image channels |
| `F` | convolution filters |
| `U` | Dense units |

Write shapes in comments and docstrings:

```text
x: (B, D)
W: (D, U)
y: (B, U)
```

For attention, show the split explicitly:

```text
(B, S, D) -> (B, S, H, d) -> (B, H, S, d) -> (B, S, D)
```

---

## Backward strategy

Use the tape by default:

- Write `forward` with `Tensor` operations.
- Do not cache inputs just for backward; the base `Layer.backward` re-runs
  `forward`.
- Implement `compute_output_shape`.

Write a manual `backward` only when:

- the explicit chain rule is the lesson, or
- replaying `forward` is awkward (e.g. a branched UNet).

If you write one, state the strategy in the class docstring and the docs page.

---

## Public exports

- `neutro.layers` re-exports every layer for convenience.
- `neutro.models` re-exports `Model`, `Sequential`, and the architecture
  factories.
- Subpackages (`neutro.layers.attention`, `neutro.autograd`) are the stable
  import paths for focused reading.
- Add a module docstring when adding a new subpackage.

---

## Documentation checklist for a new layer

1. What does it do, in one sentence?
2. What is the math?
3. What are the input/output/parameter shapes?
4. Walk through `__init__`, `build`, `forward`, and `backward` (or state that
   the base tape handles backward).
5. Show a runnable example with expected output.
6. Say which paper or idea inspired it, if any.
7. Add the page to [the docs index](README.md).
