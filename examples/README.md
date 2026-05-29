# Neutro Examples

This directory contains examples demonstrating how to use the `neutro` library for various machine learning tasks.

## 1. MNIST CNN Digit Classification
A classic convolutional neural network for classifying handwritten digits from the MNIST dataset.

**Features:**
- Data downloading and caching.
- `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers.
- `ImageDataGenerator` for real-time normalization.

**Run:**
```bash
python3 examples/mnist_cnn.py
```

## 2. WikiText-2 Transformer LLM
A character-level language model based on the Transformer architecture, trained on the WikiText-2 dataset.

**Features:**
- Character-level tokenization.
- `FlashAttention` for memory-efficient training.
- Causal masking for autoregressive modeling.
- `AdamW` optimizer.

**Run:**
```bash
python3 examples/wikitext_llm.py
```

## Requirements
Ensure you have the dependencies installed:
```bash
pip install numpy scipy
```
The examples will automatically download the necessary datasets to `~/.neutro/datasets`.
