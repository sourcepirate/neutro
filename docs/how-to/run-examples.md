# How to Run the Examples

All examples live in `examples/`. Run them from the repository root so the
`neutro` package is importable.

## First, the no-download example

```bash
python examples/00_hello_neutro.py
```

This uses synthetic data and finishes in about two seconds. If this fails,
fix your environment before trying a dataset example:
[Start Here](../START_HERE.md#if-it-fails).

## Then pick an example

| Example | What it shows | Data | Rough CPU time |
|---------|---------------|------|----------------|
| `00_hello_neutro.py` | A full training loop on synthetic data | none | seconds |
| `03_attention_tour.py` | Attention by hand, then `MultiHeadAttention` + causal mask | none | seconds |
| `04_tiny_transformer_lm.py` | Causal transformer LM, training and generation | none | seconds |
| `05_manual_backprop.py` | Manual chain rule vs the tape-backed path | none | seconds |
| `06_tokenization_tour.py` | BPE training, encoding/decoding, special tokens | none | seconds |
| `07_lora_tour.py` | LoRA parameter savings, gradients, merging | none | seconds |
| `mnist_cnn.py` | `Conv2D`, pooling, dropout, image augmentation | MNIST (download) | minutes |
| `mnist_functional_residual.py` | Functional API and a skip connection | MNIST (download) | minutes |
| `sentiment_gru.py` | Embeddings, `GRU`, padded sequences | IMDB (download) | minutes |
| `wikitext_llm.py` | Character-level transformer language model | WikiText-2 (download) | longer |
| `lora_finetune.py` | LoRA fine-tuning and weight merging | synthetic or WikiText-2 | minutes |
| `qlora_finetune.py` | 4-bit NF4 base weights + LoRA | synthetic | minutes |
| `bitnet_train.py` | BitNet block with quantized weights | WikiText-2 (download) | longer |
| `reasoning_train.py` | Tiny causal transformer on arithmetic | synthetic | minutes |
| `qwen_moe_toy.py` | Dense Qwen vs Qwen-MoE router usage | synthetic | minutes |
| `tinystories_diffusion.py` | Masked diffusion language model | synthetic | minutes |
| `mnist_vae.py` | Variational autoencoder | MNIST (download) | minutes |
| `mnist_diffusion.py` | UNet + diffusion sampling | MNIST (download) | minutes |

Dataset examples download to `~/.neutro/datasets` on first run. They may take
a few minutes depending on your connection. `mnist_cnn.py` intentionally uses
a small subset so the demo finishes quickly.

## Optional plotting dependency

`mnist_vae.py` and `mnist_diffusion.py` use `matplotlib`:

```bash
pip install matplotlib
```

The other examples need only the core dependencies.

## If an example is too slow

- Lower `epochs`, `batch_size`, or the model width at the top of the file.
- Start with `00_hello_neutro.py` and `reasoning_train.py`.
- Use `verbose=0` on `fit` to reduce progress-bar noise.
