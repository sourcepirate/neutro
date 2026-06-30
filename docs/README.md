# Neutro Documentation 📚

A line-by-line, math-first walkthrough of every component in `neutro`.  
Each entry links to the original research paper so you can read along.

---

## Table of Contents

- [Autograd Engine](#autograd-engine)
- [Layers](#layers)
  - [Attention](#attention)
  - [Convolutional](#convolutional)
  - [Core](#core)
  - [Embedding](#embedding)
  - [Normalization](#normalization)
  - [Pooling](#pooling)
  - [Recurrent](#recurrent)
  - [Transformer](#transformer)
- [Activations](#activations)
- [Models](#models)
  - [Vision](#vision)
  - [Language](#language)
- [Tokenizers](#tokenizers)
- [Optimizers](#optimizers)
- [Losses](#losses)
- [Metrics](#metrics)
- [Data & Preprocessing](#data--preprocessing)
- [Engine](#engine)
- [Callbacks](#callbacks)
- [Initializers](#initializers)
- [Utils](#utils)

---

## Autograd Engine

| Source | Doc | Research Paper |
|--------|-----|---------------|
| `neutro/autograd/tensor.py` | [tensor.md](./autograd/tensor.md) | Reverse-mode autograd — survey: [Baydin et al. (2017)](https://arxiv.org/abs/1502.05767) |
| `neutro/autograd/ops.py` | [ops.md](./autograd/ops.md) | Each primitive (matmul, softmax, etc.) is a manual backward closure |
| `neutro/autograd/function.py` | [function.md](./autograd/function.md) | `Function` API inspired by PyTorch's `torch.autograd.Function` |
| `neutro/autograd/tape.py` | — | Gradient tape — inspired by TensorFlow's `GradientTape` |
| `neutro/autograd/utils.py` | — | Broadcasting utilities for gradient propagation |
| `neutro/autograd/custom_ops.py` | — | Higher-level custom Function subclasses |

---

## Layers

### Attention

| Layer | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **BaseAttention** | `neutro/layers/attention/base_attention.py` | [base_attention.md](./layers/attention/base_attention.md) | Scaled dot-product — [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) |
| **Multi-Head Attention** | `neutro/layers/attention/mha.py` | [mha.md](./layers/attention/mha.md) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **Grouped Query Attention** | `neutro/layers/attention/gqa.py` | [gqa.md](./layers/attention/gqa.md) | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) |
| **Multi-Query Attention** | `neutro/layers/attention/mqa.py` | [mqa.md](./layers/attention/mqa.md) | [One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |
| **Multi-Head Latent Attention** | `neutro/layers/attention/mla.py` | [mla.md](./layers/attention/mla.md) | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| **FlashAttention** | `neutro/layers/attention/flash_attention.py` | [flash_attention.md](./layers/attention/flash_attention.md) | [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) / [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) |
| **PagedAttention** | `neutro/layers/attention/paged_attention.py` | [paged_attention.md](./layers/attention/paged_attention.md) | [Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180) |
| **KV Cache** | `neutro/layers/attention/kv_cache.py` | [kv_cache.md](./layers/attention/kv_cache.md) | Autoregressive decoding — [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) |

### Convolutional

| Layer | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **Conv1D** | `neutro/layers/convolutional/conv1d.py` | [conv1d.md](./layers/convolutional/conv1d.md) | [LeCun et al. (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) |
| **Conv2D** | `neutro/layers/convolutional/conv2d.py` | [conv2d.md](./layers/convolutional/conv2d.md) | [LeCun et al. (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) |

### Core

| Layer | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **Dense** | `neutro/layers/core/dense.py` | [dense.md](./layers/core/dense.md) | Standard feedforward |
| **Dropout** | `neutro/layers/core/core_utility_layers.py` | [core_utility_layers.md](./layers/core/core_utility_layers.md) | [Dropout (Srivastava et al., 2014)](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) |
| **Reparameterization** | `neutro/layers/core/reparameterization.py` | [core_utility_layers.md](./layers/core/core_utility_layers.md) | [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114) |
| **BitLinear** | `neutro/layers/core/bitlinear.py` | [bitlinear.md](./layers/core/bitlinear.md) | [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) |
| **InputLayer** | `neutro/layers/core/input_layer.py` | [input_layer.md](./layers/core/input_layer.md) | — |
| **Activation** | `neutro/layers/core/activation.py` | [activation.md](./layers/core/activation.md) | See [Activations](#activations) |
| **Flatten / Reshape** | `neutro/layers/core/core_utility_layers.py` | [core_utility_layers.md](./layers/core/core_utility_layers.md) | — |
| **Concatenate / Add** | `neutro/layers/core/merging.py` | [merging.md](./layers/core/merging.md) | [Deep Residual Learning (He et al., 2016)](https://arxiv.org/abs/1512.03385) |

### Embedding

| Layer | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **Embedding** | `neutro/layers/embedding/embedding.py` | [embedding.md](./layers/embedding/embedding.md) | [Word2Vec (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) |
| **Token & Position Embedding** | `neutro/layers/embedding/token_position_embedding.py` | [token_position_embedding.md](./layers/embedding/token_position_embedding.md) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |

### Normalization

| Layer | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **BatchNorm** | `neutro/layers/normalization/batchnorm.py` | [batchnorm.md](./layers/normalization/batchnorm.md) | [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167) |
| **LayerNorm** | `neutro/layers/normalization/layernorm.py` | [layernorm.md](./layers/normalization/layernorm.md) | [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450) |
| **RMSNorm** | `neutro/layers/normalization/normalization.py` | [normalization.md](./layers/normalization/normalization.md) | [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) |
| **GroupNorm** | `neutro/layers/normalization/normalization.py` | [normalization.md](./layers/normalization/normalization.md) | [Group Normalization (Wu & He, 2018)](https://arxiv.org/abs/1803.08494) |

### Pooling

| Layer | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **MaxPooling2D / AvgPooling2D** | `neutro/layers/pooling/pooling.py` | [pooling.md](./layers/pooling/pooling.md) | [Striving for Simplicity (Springenberg et al., 2014)](https://arxiv.org/abs/1412.6806) |
| **GlobalAvgPooling** | `neutro/layers/pooling/pooling.py` | [pooling.md](./layers/pooling/pooling.md) | [Network In Network (Lin et al., 2013)](https://arxiv.org/abs/1312.4400) |

### Recurrent

| Layer | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **LSTM** | `neutro/layers/recurrent/lstm.py` | [lstm.md](./layers/recurrent/lstm.md) | [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) |

### Transformer

| Layer | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **TransformerBlock** | `neutro/layers/transformer/transformer_block.py` | [transformer_block.md](./layers/transformer/transformer_block.md) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |

---

## Activations

| Function | Source | Doc | Research Paper |
|----------|--------|-----|---------------|
| **ReLU** | `neutro/activations/relu.py` | [activations.md](./activations/activations.md) | [Nair & Hinton (2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) |
| **Sigmoid** | `neutro/activations/sigmoid.py` | [activations.md](./activations/activations.md) | Standard |
| **Tanh** | `neutro/activations/tanh.py` | [activations.md](./activations/activations.md) | Standard |
| **Softmax** | `neutro/activations/softmax.py` | [softmax.md](./activations/softmax.md) | Standard |
| **SiLU / Swish** | `neutro/activations/silu.py` | [activations.md](./activations/activations.md) | [Swish (Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941) |

---

## Models

### Vision

| Model | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **AlexNet** | `neutro/models/vision/` | [vision_models.md](./models/vision_models.md) | [ImageNet Classification with Deep CNNs (Krizhevsky et al., 2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| **VGG16 / VGG19** | `neutro/models/vision/` | [vision_models.md](./models/vision_models.md) | [Very Deep Convolutional Networks (Simonyan & Zisserman, 2015)](https://arxiv.org/abs/1409.1556) |
| **VAE** | `neutro/models/vision/vae.py` | [vision_models.md](./models/vision_models.md) | [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114) |
| **Diffusion** | `neutro/models/vision/diffusion.py` | [vision_models.md](./models/vision_models.md) | [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) |

### Language

| Model | Source | Doc | Research Paper |
|-------|--------|-----|---------------|
| **GPT-2** | `neutro/models/language/gpt.py` | [language_models.md](./models/language_models.md) | [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://openai.com/research/better-language-models/) |
| **Llama** | `neutro/models/language/llama.py` | [language_models.md](./models/language_models.md) | [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) |
| **DeepSeek (MoE)** | `neutro/models/language/deepseek.py` | [language_models.md](./models/language_models.md) | [DeepSeek-V2 (DeepSeek-AI, 2024)](https://arxiv.org/abs/2405.04434) / [DeepSeekMoE (2024)](https://arxiv.org/abs/2401.06066) |
| **Qwen** | `neutro/models/language/qwen.py` | [language_models.md](./models/language_models.md) | [Qwen Technical Report (Bai et al., 2023)](https://arxiv.org/abs/2309.16609) |

---

## Tokenizers

| Tokenizer | Source | Doc | Research Paper |
|-----------|--------|-----|---------------|
| **RegexTokenizer** | `neutro/tokenizers/regex_tokenizer.py` | [tokenizers.md](./tokenizers/tokenizers.md) | Byte-level BPE — [GPT-2 (Radford et al., 2019)](https://openai.com/research/better-language-models/) |
| **BPETokenizer** | `neutro/tokenizers/bpe_tokenizer.py` | [tokenizers.md](./tokenizers/tokenizers.md) | [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909) |

---

## Optimizers

| Optimizer | Source | Doc | Research Paper |
|-----------|--------|-----|---------------|
| **SGD** | `neutro/optimizers/sgd.py` | [sgd.md](./optimizers/sgd.md) | Standard |
| **Adam** | `neutro/optimizers/adam.py` | [adam.md](./optimizers/adam.md) | [Adam (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980) |
| **AdamW** | `neutro/optimizers/adamw.py` | [adamw.md](./optimizers/adamw.md) | [Decoupled Weight Decay (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101) |

---

## Losses

| Loss | Source | Doc | Research Paper |
|------|--------|-----|---------------|
| **MeanSquaredError** | `neutro/losses/mse.py` | [losses.md](./losses/losses.md) | Standard |
| **CategoricalCrossentropy** | `neutro/losses/categorical_crossentropy.py` | [losses.md](./losses/losses.md) | Standard |
| **SparseCategoricalCrossentropy** | `neutro/losses/sparse_categorical_crossentropy.py` | [losses.md](./losses/losses.md) | Standard |
| **VAELoss** | `neutro/losses/vae_loss.py` | [losses.md](./losses/losses.md) | [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114) |

---

## Other Components

| Component | Source | Doc |
|-----------|--------|-----|
| **Metrics** | `neutro/metrics/` | [metrics.md](./metrics/metrics.md) |
| **Data utils** | `neutro/utils/data_utils.py` | [utils.md](./utils/utils.md) |
| **Preprocessing** | `neutro/preprocessing/` | [preprocessing.md](./preprocessing/preprocessing.md) |
| **Callbacks** | `neutro/callbacks/` | [callbacks.md](./callbacks/callbacks.md) |
| **Initializers** | `neutro/initializers/` | [initializers.md](./initializers/initializers.md) |
| **Engine (Node)** | `neutro/engine/node.py` | [node.md](./engine/node.md) |

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
