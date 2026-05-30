# 🧠 Neutro: The "Old School" Deep Learning Playground

[![codecov](https://codecov.io/gh/sourcepirate/neutro/graph/badge.svg?token=8H4Q2Q2Q2Q)](https://codecov.io/gh/sourcepirate/neutro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Neutro** is a intentionally naive, NumPy-only implementation of modern deep learning architectures. It’s the Keras experience you love, powered by the NumPy you tolerate, built specifically for people who want to peek under the hood and actually *understand* how the gears turn.

---

## 👴 The Philosophy: Why Does This Exist?

Let's be honest: modern DL frameworks are black boxes. You pip install 4GB of binaries and suddenly you're "doing AI." 

**Neutro is for the curious, the learners, and the "old-school" folks like me** who believe that if you can't build it in a matrix, you don't really know it. 

- **Learn, Don't just Run**: Every line of code is designed to be readable. We don't hide behind C++ kernels or CUDA kernels. If you want to know how FlashAttention *actually* tiles memory, you can just read the Python file.
- **A Toy, not a Tool**: This isn't meant for production. It's a playground for learning advanced algorithms (MHA, GQA, FlashAttention, LSTM) in their purest form.
- **For the Wisdom-Rich**: If you remember when 64MB of RAM was a flex and "vectorization" meant loop unrolling, this is for you. It's a fun way to play with cutting-edge 2024 algorithms using 1990s-era clarity.

---

## 🚀 What's Inside?

- **"I can't believe it's not Keras!"**: Your muscle memory is safe here. `.compile()`, `.fit()`, `.predict()`—it’s all exactly where you left it.
- **Pure NumPy Math**: We did the math so you don't have to. Every gradient, from Softmax to LSTM gates, is hand-derived and vectorized.
- **Speed (for a CPU)**: We use `im2col` for convolutions and **FlashAttention** (yes, really) to keep your CPU fans humming in a way that sounds productive.
- **Zero Heavy Dependencies**: Tired of downloading 4GB of CUDA binaries just to train on MNIST? We require exactly `numpy` and `scipy`. That’s it.

---

## 🛠 Features That'll Make You Say "Wait, You Implemented That?"

| Category | The "Fancy" Stuff | Why You Should Care |
| :--- | :--- | :--- |
| **Attention** | `FlashAttention`, `MQA`, `GQA`, `RoPE` | We have more attention variants than a distracted toddler. |
| **Tokenization** | `BPETokenizer`, `RegexTokenizer` | Byte-level BPE with regex splitting, just like the big kids. |
| **Vision** | `AlexNet`, `VGG16`, `VGG19`, `im2col` | Classical and modern vision architectures, vectorized. |
| **LLMs** | `Llama`, `Qwen`, `DeepSeek` (MoE) | Yes, you can run a (very tiny) MoE model on your CPU. |
| **Modern Ops** | `RMSNorm`, `SiLU`, `SwiGLU` | The secret sauce of modern LLMs, hand-implemented. |
| **Optimizers** | `AdamW`, `Adam`, `SGD+Momentum` | Keep your weights from exploding like a bad science fair project. |

---

## 🏆 The Hall of Fame: Pre-built Architectures

Why build from scratch when we've already done the heavy lifting?

- **The Visionaries**: `AlexNet`, `VGG16`, `VGG19`
- **The Linguists**: `GPT-2`, `LlamaTiny`, `QwenTiny`, `DeepSeekTiny` (Mixture of Experts)

---

## 💻 Show Me The Code!

If you know Keras, you already know Neutro. It's that simple.

```python
from neutro.models import Sequential
from neutro.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build a CNN that actually fits in your head
model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile it like it's 2015
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit it like a tailored suit
model.fit(train_flow, epochs=10)
```

---

## 📂 Deep Dives & Nerdy Stuff

We documented everything because we know you like to check the math:

- [**Attention Mechanisms**](./docs/layers/attention/) - How we made FlashAttention work on a CPU.
- [**Convolutional Magic**](./docs/layers/convolutional/) - The `im2col` deep dive.
- [**Activations & Gradients**](./docs/activations/) - Proofs for the brave.
- [**Optimizers**](./docs/optimizers/) - Why AdamW is better than your ex.

---

## 🧪 Examples to Flex Your CPU

Check out the `examples/` folder for end-to-end scripts:
- `mnist_cnn.py`: Standard digit classification with real-time augmentation.
- `wikitext_llm.py`: A character-level Transformer that actually talks back.

---

## 🏗 Installation

```bash
git clone https://github.com/sourcepirate/neutro.git
cd neutro
pip install -e .
```

---

**Disclaimer**: This is a hobby project for learning and exploration. It is intentionally naive, likely inefficient compared to compiled kernels, and 100% focused on the joy of understanding advanced algorithms. If you're looking to change the world with AGI, go to PyTorch. If you're looking to understand why your Transformer works while drinking a nice cup of tea, you're in the right place.
