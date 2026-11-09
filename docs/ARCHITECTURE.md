# Architecture Map

This page is the map you keep beside you while reading the code. It answers
three questions:

1. Where does a piece of functionality live?
2. What happens when I call `model.fit(...)`?
3. What does a shape like `(B, S, D)` mean?

---

## Folder map

```text
neutro/
  activations/ ReLU, Sigmoid, Tanh, Softmax, SiLU
  autograd/ Tensor, GradientTape, primitive ops, custom Function ops
  callbacks/ History, EarlyStopping, ModelCheckpoint, LR schedules
  engine/ KerasTensor and Node for the Functional API
  initializers/ Glorot, He, zeros, random normal
  layers/
    attention/ MHA, MQA, GQA, MLA, FlashAttention-shaped API, KV cache
    convolutional/ Conv1D, Conv2D (im2col)
    core/ Dense, Dropout, Flatten, activations, merging, MoE, LoRA
    diffusion/ AdaLN, DiT block, time embedding, mask scheduler
    embedding/ Embedding, token/position embedding, time embedding
    normalization/ LayerNorm, RMSNorm, BatchNorm, GroupNorm, QKNorm
    pooling/ MaxPooling2D, global pooling, upsampling
    recurrent/ SimpleRNN, GRU, LSTM
    transformer/ TransformerBlock, MoE block, Qwen block, BitNet block
  losses/ MSE, categorical/sparse CE, masked CE, VAE loss
  metrics/ Accuracy, sparse accuracy, precision, recall, F1
  models/ Model, Sequential, training loop, summary, and factories
  optimizers/ SGD, Adam, AdamW, LR schedules
  preprocessing/ image augmentation, text tokenizer, sequence padding
  tokenizers/ byte-level BPE and regex BPE
  utils/ convolution helpers, data download, RoPE, visualization
```

**Start with** `layers/core/dense.py`, `layers/base.py`,
`models/base_model.py`, and `autograd/tape.py`. Everything else is a variation
on those ideas.

---

## What happens during `model.fit(...)`

```text
fit(x, y)
  |
  v
Model._fit_loop
  | shuffle, slice batches, call callbacks
  v
Model._train_on_batch
  |
  +--> Model.forward
  | |
  | +--> for each layer: Layer.__call__ -> build (first time) -> forward
  |
  +--> loss_fn(y, output)
  |
  +--> Model.backward(loss_gradient)
  | |
  | +--> for each layer in reverse: Layer.backward(grad_output)
  | |
  | +--> re-run forward inside a GradientTape
  | +--> tape.gradient(probe_loss, params)
  | +--> copy Tensor.grad into layer.grads
  |
  +--> optimizer.step(all_trainable_layers)
         |
         +--> for each layer, each param: param -= lr * grad (or Adam rule)
```

The key trick: `Layer.backward` is a **generic tape fallback**. It does not
need a hand-written derivative for each layer. It replays the layer's
`forward` with the captured inputs and parameters watched, builds the scalar
`(output * grad_output).sum`, and asks the tape for gradients. That is why
`Dense` can be three lines of math and still train.

A few advanced layers (`BitLinear`, `LoRA`, `QLoRA`, `UNet`, `VAE`,
`MaskedDiffusionLM`) implement `backward` by hand. See
[Implementation status](IMPLEMENTATION_STATUS.md#backward-strategies).

---

## Shape notation

The code uses the same letters everywhere. Read `(B, S, D)` as:

| Symbol | Usually means | Example |
|--------|---------------|---------|
| `B` | batch size | 32 sentences |
| `S` or `T` | sequence length | 128 tokens |
| `D` | model/embedding width | 512 |
| `H` | number of attention heads | 8 |
| `d` | per-head width, usually `D / H` | 64 |
| `H_img`, `W_img` | image height and width | 28, 28 |
| `V` | vocabulary size | 50257 |
| `C` | image channels | 3 |
| `F` | convolution filters | 64 |
| `U` | Dense output units | 10 |

Examples:

- `Dense`: `(B, D) @ (D, U) -> (B, U)`
- `Conv2D`: `(B, H_img, W_img, C) -> (B, H_img', W_img', F)`
- `Embedding`: `(B, S)` integer ids -> `(B, S, D)`
- Attention: `(B, S, D) -> (B, H, S, d)` inside, then `(B, S, D)` out
- LM head: `(B, S, D) @ (D, V) -> (B, S, V)`

---

## Reading paths

**"I want to train a model."**
`layers/core/dense.py` → `models/base_model.py` → `optimizers/adam.py` →
`losses/categorical_crossentropy.py`.

**"I want to understand backprop."**
`autograd/tensor.py` → `autograd/tape.py` → `layers/base.py` (`backward`) →
`autograd/ops.py`.

**"I want to understand attention."**
`layers/attention/base_attention.py` → `layers/attention/mha.py` →
`layers/attention/flash_attention.py` → `layers/transformer/transformer_block.py`.

**"I want to understand modern LLMs."**
`layers/normalization/rmsnorm.py` → `layers/core/swiglu_mlp.py` →
`layers/attention/gqa.py` → `layers/attention/mla.py` →
`models/language/llama.py` and `models/language/deepseek.py`.

**"I want to understand convolutions."**
`utils/conv_utils.py` (`im2col`) → `autograd/custom_ops.py` (`Conv2DFunction`) →
`layers/convolutional/conv2d.py`.

---

## Documentation layout

- **Learn:** [START_HERE](START_HERE.md), [tutorials](tutorials/01-first-model.md)
- **Do a task:** [how-to guides](how-to/run-examples.md)
- **Understand a design:** the deep dives under `layers/`, `autograd/`, and
  `models/`
- **Look up an API:** [docs index](README.md), [API conventions](API_CONVENTIONS.md),
  and docstrings

If a page reads like reference but starts with a story, it is probably an
explanation page. If it reads like a lesson, it is a tutorial. Keeping those
separate is deliberate: it is easier to find what you need.
