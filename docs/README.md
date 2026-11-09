# Neutro Documentation

A line-by-line, math-first walkthrough of every component in `neutro`, plus a
guided path for getting started. Each deep dive links to the original research
paper where one exists.

---

## New here? Read these first

| Step | Page | What it covers |
|------|------|----------------|
| 1 | [Start Here](START_HERE.md) | Install, first run, troubleshooting |
| 2 | [Tutorial 01: Your first model](tutorials/01-first-model.md) | `Dense`, shapes, `fit`, a real gradient |
| 3 | [Tutorial 02: Gradients and the tape](tutorials/02-gradients-and-the-tape.md) | How one training step works |
| 4 | [Tutorial 03: Attention from scratch](tutorials/03-attention-from-scratch.md) | Q/K/V, softmax, heads, causal masks |
| 5 | [Tutorial 04: A tiny transformer LM](tutorials/04-tiny-transformer-lm.md) | Training and generating with a causal transformer |
| 6 | [Tutorial 05: Manual backprop](tutorials/05-manual-backprop.md) | Chain rule by hand vs the tape |
| 7 | [Tutorial 06: Tokenization](tutorials/06-tokenization.md) | Bytes, BPE, special tokens, save/load |
| 8 | [Tutorial 07: LoRA fine-tuning](tutorials/07-lora-finetuning.md) | Frozen base + low-rank adapter |
| 9 | [Architecture map](ARCHITECTURE.md) | Folder map, call flow, shape notation |
| 10 | [Glossary](GLOSSARY.md) | Tensor, parameter, logits, head, MoE, LoRA, … |
| 11 | [API conventions](API_CONVENTIONS.md) | Parameter names, attributes, shapes, exports |
| 12 | [Implementation status](IMPLEMENTATION_STATUS.md) | Implemented vs simplified vs alias |

**How-to guides**

- [Run the examples](how-to/run-examples.md)
- [Train on MNIST](how-to/train-mnist.md)

The rest of this page is the reference index. You do not need to read it in
order.

---

## Autograd Engine

| Source | Doc | Notes |
|--------|-----|-------|
| `neutro/autograd/tensor.py` | [tensor.md](./autograd/tensor.md) | `Tensor`: data + gradient |
| `neutro/autograd/tape.py` | [tape.md](./autograd/tape.md) | `GradientTape`: record and reverse |
| `neutro/autograd/ops.py` | [ops.md](./autograd/ops.md) | Primitive ops and backward rules |
| `neutro/autograd/function.py` | [function.md](./autograd/function.md) | Custom `Function` ops |
| `neutro/autograd/custom_ops.py` | — | Conv, pooling, gather, quantize ops |
| `neutro/autograd/utils.py` | — | Broadcast backward, numeric gradient check |

Overview: [autograd.md](./autograd/autograd.md).

---

## Layers

### Attention

| Layer | Source | Doc | Paper |
|-------|--------|-----|-------|
| **BaseAttention** | `neutro/layers/attention/base_attention.py` | [base_attention.md](./layers/attention/base_attention.md) | [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) |
| **Multi-Head Attention** | `neutro/layers/attention/mha.py` | [mha.md](./layers/attention/mha.md) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **Grouped Query Attention** | `neutro/layers/attention/gqa.py` | [gqa.md](./layers/attention/gqa.md) | [GQA](https://arxiv.org/abs/2305.13245) |
| **Multi-Query Attention** | `neutro/layers/attention/mqa.py` | [mqa.md](./layers/attention/mqa.md) | [One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |
| **Multi-Head Latent Attention** | `neutro/layers/attention/mla.py` | [mla.md](./layers/attention/mla.md) | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| **FlashAttention** | `neutro/layers/attention/flash_attention.py` | [flash_attention.md](./layers/attention/flash_attention.md) | [FlashAttention](https://arxiv.org/abs/2205.14135) — **API-compatible, simplified implementation** |
| **PagedAttention** | `neutro/layers/attention/paged_attention.py` | [paged_attention.md](./layers/attention/paged_attention.md) | [PagedAttention](https://arxiv.org/abs/2309.06180) |
| **KV Cache** | `neutro/layers/attention/kv_cache.py` | [kv_cache.md](./layers/attention/kv_cache.md) | Autoregressive decoding |

### Convolutional

| Layer | Source | Doc | Paper |
|-------|--------|-----|-------|
| **Conv1D** | `neutro/layers/convolutional/conv1d.py` | [conv1d.md](./layers/convolutional/conv1d.md) | [LeCun et al. (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) |
| **Conv2D** | `neutro/layers/convolutional/conv2d.py` | [conv2d.md](./layers/convolutional/conv2d.md) | [LeCun et al. (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) |

### Core

| Layer | Source | Doc | Paper |
|-------|--------|-----|-------|
| **Dense** | `neutro/layers/core/dense.py` | [dense.md](./layers/core/dense.md) | Standard feedforward |
| **Dropout** | `neutro/layers/core/dropout.py` | [core_utility_layers.md](./layers/core/core_utility_layers.md) | [Srivastava et al. (2014)](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) |
| **Flatten** | `neutro/layers/core/flatten.py` | [core_utility_layers.md](./layers/core/core_utility_layers.md) | — |
| **Input / InputLayer** | `neutro/layers/core/input_layer.py` | [input_layer.md](./layers/core/input_layer.md) | — |
| **Activation** | `neutro/layers/core/activation.py` | [activation.md](./layers/core/activation.md) | See [Activations](#activations) |
| **BitLinear** | `neutro/layers/core/bitlinear.py` | [bitlinear.md](./layers/core/bitlinear.md) | [BitNet](https://arxiv.org/abs/2310.11453) |
| **Reparameterization** | `neutro/layers/core/reparameterization.py` | [core_utility_layers.md](./layers/core/core_utility_layers.md) | [VAE](https://arxiv.org/abs/1312.6114) |
| **LoRA** | `neutro/layers/core/lora.py` | [lora.md](./layers/core/lora.md) | [LoRA](https://arxiv.org/abs/2106.09685) |
| **QLoRA** | `neutro/layers/core/qlora.py` | [qlora.md](./layers/core/qlora.md) | [QLoRA](https://arxiv.org/abs/2305.14314) |
| **MoE** | `neutro/layers/core/moe.py` | [moe.md](./layers/core/moe.md) | [Switch Transformer](https://arxiv.org/abs/2101.03961) |
| **SwiGLU MLP** | `neutro/layers/core/swiglu_mlp.py` | [swiglu_mlp.md](./layers/core/swiglu_mlp.md) | [GLU Variants](https://arxiv.org/abs/2002.05202) |
| **Add / Concatenate / Multiply / Average / Maximum / Minimum** | `neutro/layers/core/merging.py` | [merging.md](./layers/core/merging.md) | [Residual Learning](https://arxiv.org/abs/1512.03385) |

### Embedding

| Layer | Source | Doc | Paper |
|-------|--------|-----|-------|
| **Embedding** | `neutro/layers/embedding/embedding.py` | [embedding.md](./layers/embedding/embedding.md) | [Word2Vec](https://arxiv.org/abs/1301.3781) |
| **Token & Position Embedding** | `neutro/layers/embedding/token_position_embedding.py` | [token_position_embedding.md](./layers/embedding/token_position_embedding.md) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **TimeEmbedding** | `neutro/layers/embedding/time_embedding.py` | — | Diffusion timestep conditioning |

### Normalization

| Layer | Source | Doc | Paper |
|-------|--------|-----|-------|
| **BatchNorm** | `neutro/layers/normalization/batchnorm.py` | [batchnorm.md](./layers/normalization/batchnorm.md) | [Batch Normalization](https://arxiv.org/abs/1502.03167) |
| **LayerNorm** | `neutro/layers/normalization/layernorm.py` | [layernorm.md](./layers/normalization/layernorm.md) | [Layer Normalization](https://arxiv.org/abs/1607.06450) |
| **RMSNorm** | `neutro/layers/normalization/rmsnorm.py` | [normalization.md](./layers/normalization/normalization.md) | [RMSNorm](https://arxiv.org/abs/1910.07467) |
| **GroupNorm** | `neutro/layers/normalization/groupnorm.py` | [normalization.md](./layers/normalization/normalization.md) | [Group Normalization](https://arxiv.org/abs/1803.08494) |
| **QKNorm** | `neutro/layers/normalization/qknorm.py` | [qknorm.md](./layers/normalization/qknorm.md) | Query/key normalization before attention |

### Pooling

| Layer | Source | Doc |
|-------|--------|-----|
| **MaxPooling2D** | `neutro/layers/pooling/maxpooling2d.py` | [pooling.md](./layers/pooling/pooling.md) |
| **GlobalAveragePooling2D / GlobalMaxPooling2D** | `neutro/layers/pooling/global_pooling.py` | [pooling.md](./layers/pooling/pooling.md) |
| **UpSampling2D** | `neutro/layers/pooling/upsampling2d.py` | [pooling.md](./layers/pooling/pooling.md) |

> There is no `AvgPooling2D` in the current implementation. Use
> `GlobalAveragePooling2D` for global average pooling, or a `Conv2D` with a
> fixed averaging kernel.

### Recurrent

| Layer | Source | Doc | Paper |
|-------|--------|-----|-------|
| **SimpleRNN** | `neutro/layers/recurrent/simple_rnn.py` | [recurrent.md](./layers/recurrent/recurrent.md) | Standard Elman RNN |
| **GRU** | `neutro/layers/recurrent/gru.py` | [recurrent.md](./layers/recurrent/recurrent.md) | [Cho et al. (2014)](https://arxiv.org/abs/1406.1078) |
| **LSTM** | `neutro/layers/recurrent/lstm.py` | [lstm.md](./layers/recurrent/lstm.md) | [Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) |

### Transformer and diffusion blocks

| Layer | Source | Doc |
|-------|--------|-----|
| **TransformerBlock** | `neutro/layers/transformer/transformer_block.py` | [transformer_block.md](./layers/transformer/transformer_block.md) |
| **MoETransformerBlock** | `neutro/layers/transformer/moe_block.py` | [moe_block.md](./layers/transformer/moe_block.md) |
| **QwenBlock** | `neutro/layers/transformer/qwen_block.py` | [qwen_block.md](./layers/transformer/qwen_block.md) |
| **BitNetBlock** | `neutro/layers/transformer/bitnet_block.py` | — |
| **AdaLN / AdaLNZero** | `neutro/layers/diffusion/adaln.py` | [diffusion.md](./layers/diffusion/diffusion.md) |
| **DiTBlock** | `neutro/layers/diffusion/dit_block.py` | [diffusion.md](./layers/diffusion/diffusion.md) |
| **TimeMLPEmbedding** | `neutro/layers/diffusion/time_mlp.py` | [diffusion.md](./layers/diffusion/diffusion.md) |
| **DiscreteDiffusionScheduler** | `neutro/layers/diffusion/discrete_scheduler.py` | [diffusion.md](./layers/diffusion/diffusion.md) |

---

## Activations

| Function | Source | Doc | Paper |
|----------|--------|-----|-------|
| **ReLU** | `neutro/activations/relu.py` | [activations.md](./activations/activations.md) | [Nair & Hinton (2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) |
| **Sigmoid** | `neutro/activations/sigmoid.py` | [activations.md](./activations/activations.md) | Standard |
| **Tanh** | `neutro/activations/tanh.py` | [activations.md](./activations/activations.md) | Standard |
| **Softmax** | `neutro/activations/softmax.py` | [softmax.md](./activations/softmax.md) | Standard |
| **SiLU / Swish** | `neutro/activations/silu.py` | [activations.md](./activations/activations.md) | [Swish](https://arxiv.org/abs/1710.05941) |

---

## Models

### Vision

| Model | Source | Doc |
|-------|--------|-----|
| **AlexNet** | `neutro/models/vision/alexnet.py` | [vision_models.md](./models/vision_models.md) |
| **VGG16 / VGG19** | `neutro/models/vision/vgg.py` | [vision_models.md](./models/vision_models.md) |
| **VAE** | `neutro/models/vision/vae.py` | [vision_models.md](./models/vision_models.md) |
| **UNet** | `neutro/models/vision/unet.py` | [vision_models.md](./models/vision_models.md) |
| **DiffusionModel** | `neutro/models/vision/diffusion_model.py` | [vision_models.md](./models/vision_models.md) |

### Language

| Model | Source | Doc |
|-------|--------|-----|
| **GPT1 / GPT2** (`GPT1Tiny` / `GPT2Tiny` aliases) | `neutro/models/language/gpt.py` | [language_models.md](./models/language_models.md) |
| **LlamaTiny** | `neutro/models/language/llama.py` | [language_models.md](./models/language_models.md) |
| **Qwen Tiny variants** | `neutro/models/language/qwen.py` | [language_models.md](./models/language_models.md) |
| **DeepSeek Tiny variants** | `neutro/models/language/deepseek.py` | [language_models.md](./models/language_models.md) |
| **BitNetTiny** | `neutro/models/language/bitnet.py` | — |
| **MaskedDiffusionLM** | `neutro/models/language/diffusion_lm.py` | [diffusion_lm.md](./models/language/diffusion_lm.md) |

> Several language factories are educational variants or aliases. See
> [Implementation status](./IMPLEMENTATION_STATUS.md#language-models) before
> treating a name as the full published architecture.

### Model API

| Component | Source | Doc |
|-----------|--------|-----|
| **Model / Sequential** | `neutro/models/base_model.py` | [model.md](./models/model.md) |
| **Functional API** | `neutro/engine/node.py` | [node.md](./engine/node.md) |

---

## Tokenizers

| Tokenizer | Source | Doc |
|-----------|--------|-----|
| **BPETokenizer** | `neutro/tokenizers/bpe.py` | [tokenizers.md](./tokenizers/tokenizers.md) |
| **RegexTokenizer** | `neutro/tokenizers/bpe.py` | [tokenizers.md](./tokenizers/tokenizers.md) |
| **tiktoken compatibility** | `neutro/tokenizers/tiktoken_compat.py` | [tokenizers.md](./tokenizers/tokenizers.md) |

---

## Optimizers

| Optimizer | Source | Doc |
|-----------|--------|-----|
| **SGD** | `neutro/optimizers/sgd.py` | [sgd.md](./optimizers/sgd.md) |
| **Adam** | `neutro/optimizers/adam.py` | [adam.md](./optimizers/adam.md) |
| **AdamW** | `neutro/optimizers/adamw.py` | [adamw.md](./optimizers/adamw.md) |
| **Schedules** | `neutro/optimizers/schedules.py` | [optimizers.md](./optimizers/optimizers.md) |

`model.compile(optimizer="adam",...)` and `model.compile(optimizer=Adam,...)`
are both supported.

---

## Losses

| Loss | Source | Doc |
|------|--------|-----|
| **MeanSquaredError** | `neutro/losses/mse.py` | [losses.md](./losses/losses.md) |
| **CategoricalCrossentropy** | `neutro/losses/categorical_crossentropy.py` | [losses.md](./losses/losses.md) |
| **SparseCategoricalCrossentropy** | `neutro/losses/sparse_categorical_crossentropy.py` | [losses.md](./losses/losses.md) |
| **MaskedSparseCrossentropy** | `neutro/losses/masked_sparse_ce.py` | [masked_sparse_ce.md](./losses/masked_sparse_ce.md) |
| **VAELoss** | `neutro/losses/vae_loss.py` | [losses.md](./losses/losses.md) |

---

## Metrics

| Metric | Source | Doc |
|--------|--------|-----|
| **Accuracy** | `neutro/metrics/accuracy.py` | [metrics.md](./metrics/metrics.md) |
| **SparseAccuracy** | `neutro/metrics/sparse_accuracy.py` | [metrics.md](./metrics/metrics.md) |
| **Precision / Recall / F1** | `neutro/metrics/precision.py`, `recall.py`, `f1_score.py` | [metrics.md](./metrics/metrics.md) |

---

## Data, Preprocessing, and Engine

| Component | Source | Doc |
|-----------|--------|-----|
| **DataLoader** | `neutro/data.py` | [data.md](./data/data.md) |
| **ImageDataGenerator** | `neutro/preprocessing/image.py` | [preprocessing.md](./preprocessing/preprocessing.md) |
| **Tokenizer / pad_sequences** | `neutro/preprocessing/text.py`, `sequence.py` | [preprocessing.md](./preprocessing/preprocessing.md) |
| **KerasTensor / Node** | `neutro/engine/node.py` | [node.md](./engine/node.md) |
| **Callbacks** | `neutro/callbacks/` | [callbacks.md](./callbacks/callbacks.md) |
| **Initializers** | `neutro/initializers/` | [initializers.md](./initializers/initializers.md) |
| **Utils** | `neutro/utils/` | [utils.md](./utils/utils.md) |

---

## References

- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2017). **Automatic Differentiation in Machine Learning: a Survey**. *JMLR*. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). **Gradient-Based Learning Applied to Document Recognition**. *Proceedings of the IEEE*. [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- Vaswani, A., et al. (2017). **Attention Is All You Need**. *NeurIPS*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Shazeer, N. (2019). **Fast Transformer Decoding: One Write-Head is All You Need**. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Dao, T., et al. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. *NeurIPS*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, T. (2023). **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Ainslie, J., et al. (2023). **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**. *EMNLP*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Kwon, W., et al. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention**. *SOSP*. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- DeepSeek-AI. (2024). **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI. (2024). **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- Touvron, H., et al. (2023). **Llama 2: Open Foundation and Fine-Tuned Chat Models**. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- Bai, J., et al. (2023). **Qwen Technical Report**. [arXiv:2309.16609](https://arxiv.org/abs/2309.16609)
- Radford, A., et al. (2019). **Language Models are Unsupervised Multitask Learners**. [OpenAI](https://openai.com/research/better-language-models/)
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). **ImageNet Classification with Deep Convolutional Neural Networks**. *NeurIPS*. [Paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- Simonyan, K., & Zisserman, A. (2015). **Very Deep Convolutional Networks for Large-Scale Image Recognition**. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- Ioffe, S., & Szegedy, C. (2015). **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**. *ICML*. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). **Layer Normalization**. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Zhang, B., & Sennrich, R. (2019). **Root Mean Square Layer Normalization**. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Wu, Y., & He, K. (2018). **Group Normalization**. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
- Kingma, D. P., & Welling, M. (2014). **Auto-Encoding Variational Bayes**. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Srivastava, N., et al. (2014). **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**. *JMLR*. [PDF](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
- Ramachandran, P., Zoph, B., & Le, Q. V. (2017). **Swish: a Self-Gated Activation Function**. [arXiv:1710.05941](https://arxiv.org/abs/1710.05941)
- Wang, H., et al. (2023). **BitNet: Scaling 1-bit Transformers for Large Language Models**. [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)
- Sennrich, R., Haddow, B., & Birch, A. (2016). **Neural Machine Translation of Rare Words with Subword Units**. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Loshchilov, I., & Hutter, F. (2017). **Decoupled Weight Decay Regularization**. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Kingma, D. P., & Ba, J. (2014). **Adam: A Method for Stochastic Optimization**. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- He, K., et al. (2016). **Deep Residual Learning for Image Recognition**. *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*. [PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)
- Nair, V., & Hinton, G. E. (2010). **Rectified Linear Units Improve Restricted Boltzmann Machines**. [PDF](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)
- Hu, E. J., et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers, T., et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
