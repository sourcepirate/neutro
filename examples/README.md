# Neutro Examples

Start with `00_hello_neutro.py`. It needs no downloads and finishes in about
two seconds. After that, pick an example based on what you want to learn.

All examples are run from the repository root:

```bash
python examples/00_hello_neutro.py
```

Dataset examples download to `~/.neutro/datasets` on first run. A few plotting
examples also need `matplotlib`.

## Learning path

| # | Example | What it teaches | Data | Rough time |
|---|---------|-----------------|------|------------|
| 00 | `00_hello_neutro.py` | Forward, loss, backward, optimizer | synthetic | seconds |
| 01 | `03_attention_tour.py` | Attention by hand, then `MultiHeadAttention` + causal mask | synthetic | seconds |
| 02 | `04_tiny_transformer_lm.py` | Causal transformer LM, training and generation | synthetic | seconds |
| 03 | `05_manual_backprop.py` | Manual chain rule vs the tape-backed path | synthetic | seconds |
| 04 | `mnist_cnn.py` | `Conv2D`, pooling, dropout, image rescaling | MNIST | minutes |
| 05 | `mnist_functional_residual.py` | Functional API, skip connections, DAG backward | MNIST | minutes |
| 06 | `06_tokenization_tour.py` | BPE training, encoding/decoding, special tokens | synthetic | seconds |
| 07 | `07_lora_tour.py` | LoRA parameter savings, gradients, merging | synthetic | seconds |
| 08 | `sentiment_gru.py` | Embeddings, `GRU`, padded sequences, text classification | IMDB | minutes |
| 09 | `reasoning_train.py` | Tiny causal transformer on arithmetic | synthetic | minutes |
| 10 | `wikitext_llm.py` | Character-level transformer LM | WikiText-2 | longer |
| 11 | `qwen_moe_toy.py` | Dense Qwen vs Qwen-MoE, router utilization | synthetic | minutes |
| 12 | `lora_finetune.py` | LoRA adapters and weight merging | synthetic/WikiText-2 | minutes |
| 13 | `qlora_finetune.py` | NF4 base weights + LoRA | synthetic | minutes |
| 14 | `bitnet_train.py` | 1-bit / 1.58-bit BitNet blocks | WikiText-2 | longer |
| 15 | `tinystories_diffusion.py` | Masked diffusion LM with a DiT backbone | synthetic | minutes |
| 16 | `mnist_vae.py` | Variational autoencoder | MNIST | minutes |
| 17 | `mnist_diffusion.py` | UNet + diffusion sampling (needs `matplotlib`) | MNIST | minutes |

## What each one does

### 00 — Hello Neutro
A two-layer MLP on synthetic 8-feature, 3-class data. Prints the model summary
and a loss that goes down. This is the example to read first.

### 01 — Attention tour
Computes attention by hand with NumPy, then uses `MultiHeadAttention` and a
causal mask. Companion to
[Tutorial 03](../docs/tutorials/03-attention-from-scratch.md).

### 02 — Tiny transformer LM
Trains a causal transformer on a synthetic next-token task, checks predictions,
and generates new tokens. Companion to
[Tutorial 04](../docs/tutorials/04-tiny-transformer-lm.md).

### 03 — Manual backprop
Implements a Dense layer's chain rule by hand and compares `dW`, `db`, and
`dX` with the tape-backed implementation. Companion to
[Tutorial 05](../docs/tutorials/05-manual-backprop.md).

### 04 — MNIST CNN
A convolutional classifier with `Conv2D`, `MaxPooling2D`, `Flatten`,
`Dense`, and `Dropout`. Uses a 1,000-image subset so it finishes quickly. See
[How to train on MNIST](../docs/how-to/train-mnist.md).

### 05 — Functional residual model
Builds a ResNet-style skip connection with `Input`, `Add`, and
`Model(inputs, outputs)`. Demonstrates the Functional API and backward through
a DAG.

### 06 — Tokenization tour
Trains a byte-level BPE tokenizer, encodes/decodes text, shows special tokens,
round-trips Unicode, and saves/loads the vocabulary. Companion to
[Tutorial 06](../docs/tutorials/06-tokenization.md).

### 07 — LoRA tour
Compares a Dense layer with `LoRADense`, shows the trainable parameter savings,
checks the zero-initialized adapter, and merges the adapter back into the base
weight. Companion to [Tutorial 07](../docs/tutorials/07-lora-finetuning.md).

### 08 — Sentiment GRU
Embeds IMDB reviews, runs a `GRU`, and classifies sentiment. Shows
`pad_sequences` and embedding-based text input.

### 09 — Reasoning transformer
A character-level causal transformer trained to complete `Q:a+b=?A:r`.
Small enough to train on a laptop CPU.

### 10 — WikiText LLM
A character-level transformer language model trained on WikiText-2. Uses
`FlashAttention` (the simplified attention implementation; see
[Implementation status](../docs/IMPLEMENTATION_STATUS.md)) and causal masking.

### 11 — Qwen dense vs MoE
Trains `Qwen2Tiny` and `Qwen3MoETiny` on the same toy task and prints router
utilization.

### 12 — LoRA fine-tuning
Pre-trains a small transformer, adapts it with `apply_lora`, fine-tunes only
the LoRA parameters, merges the weights, and generates text.

### 13 — QLoRA fine-tuning
The same workflow with `QLoRADense` (NF4 base weights + double quantization).

### 14 — BitNet
Trains a BitNet-style model with quantized weights and activations.

### 15 — TinyStories diffusion
Trains a `MaskedDiffusionLM` with absorbing-mask diffusion and confidence-based
unmasking.

### 16 — MNIST VAE
Trains a variational autoencoder with a reconstruction + KL loss.

### 17 — MNIST diffusion
Trains a time-conditioned UNet on MNIST and samples with a Gaussian diffusion
process.

## Requirements

Core examples need only the library dependencies. Plotting examples also need:

```bash
pip install matplotlib
```

If an example is too slow, lower `epochs`, `batch_size`, or the model width at
the top of the file. See [How to run examples](../docs/how-to/run-examples.md).
