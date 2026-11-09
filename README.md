# 🧠 Neutro: The "Old School" Deep Learning Playground

[![codecov](https://codecov.io/gh/sourcepirate/neutro/graph/badge.svg?token=8H4Q2Q2Q2Q)](https://codecov.io/gh/sourcepirate/neutro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Neutro** is an intentionally naive, NumPy-only implementation of modern deep
learning. It keeps the Keras API you know (`.compile()`, `.fit()`, `.predict()`)
so you can spend your attention on the algorithms instead of the boilerplate.

It is built for people who want to peek under the hood and understand how the
gears turn. Every gradient, convolution, attention head, and optimizer update
is Python you can read.

---

## 🚀 Start Here in 5 Minutes

### 1. Install

```bash
git clone https://github.com/sourcepirate/neutro.git
cd neutro
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

Neutro targets Python 3.10+. The core dependencies are `numpy`, `scipy`,
`joblib`, `tqdm`, and `regex`; `pytest` is included with `.[test]`. A couple of
plotting examples also use `matplotlib`.

### 2. Run one model

```bash
python examples/00_hello_neutro.py
```

You should see a model summary and a loss that goes down:

```text
Loss after epoch 1: 1.0617
Loss after epoch 5: 0.3720
```

That is the whole loop:

```text
forward pass -> loss -> backward pass -> optimizer step
```

### 3. Run the tests

```bash
pytest -q
```

### 4. Follow the guided path

| Read | What you get |
|------|--------------|
| [Start Here](docs/START_HERE.md) | Install, first run, troubleshooting |
| [Tutorial 01: Your first model](docs/tutorials/01-first-model.md) | Shapes, `Dense`, `fit`, and a real gradient |
| [Tutorial 02: Gradients and the tape](docs/tutorials/02-gradients-and-the-tape.md) | How one training step works |
| [Tutorial 03: Attention from scratch](docs/tutorials/03-attention-from-scratch.md) | Q/K/V, softmax, heads, causal masks |
| [Tutorial 04: A tiny transformer LM](docs/tutorials/04-tiny-transformer-lm.md) | Training and generating with a causal transformer |
| [Tutorial 05: Manual backprop](docs/tutorials/05-manual-backprop.md) | Chain rule by hand vs the tape |
| [Tutorial 06: Tokenization](docs/tutorials/06-tokenization.md) | Bytes, BPE, special tokens, save/load |
| [Tutorial 07: LoRA fine-tuning](docs/tutorials/07-lora-finetuning.md) | Freeze the base, train a low-rank update |
| [Architecture map](docs/ARCHITECTURE.md) | Where everything lives |
| [Glossary](docs/GLOSSARY.md) | Tensor, parameter, logits, head, MoE, LoRA, … |
| [Implementation status](docs/IMPLEMENTATION_STATUS.md) | What is real, simplified, or an alias |

The full documentation index is [docs/README.md](docs/README.md). To preview
the documentation site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## 👴 The Philosophy: Why Does This Exist?

Modern DL frameworks are black boxes. You `pip install` 4 GB of binaries and
suddenly you're "doing AI."

**Neutro is for the curious, the learners, and the "old-school" folks** who
believe that if you can't build it in a matrix, you don't really know it.

- **Learn, don't just run**: every line is meant to be readable. If you want to
  know how attention tiles memory, or why AdamW decouples weight decay, you can
  read the Python file.
- **A toy, not a tool**: this is a playground for learning algorithms
  (MHA, GQA, FlashAttention-shaped attention, LSTM, MoE, LoRA, diffusion), not
  a production framework.
- **For the wisdom-rich**: if you remember when 64 MB of RAM was a flex, this
  is a fun way to play with modern algorithms using 1990s-era clarity.

---

## 🔄 Two Ways to Understand Backprop

Neutro is its own autograd engine. There are two paths, and the beginner path
is the first one:

1. **The tape (default).** Most layers write only `forward`. The base
   `Layer.backward` re-runs that forward pass inside a `GradientTape` and asks
   the tape for gradients. This keeps layer code short and composable.
2. **Manual backprop (advanced).** A few layers (`BitLinear`, `LoRA`, `QLoRA`,
   `UNet`, `VAE`, `MaskedDiffusionLM`) write `backward` by hand with explicit
   chain-rule math. This is where the gradient mechanics are the lesson.

Both paths are documented in [Implementation status](docs/IMPLEMENTATION_STATUS.md#backward-strategies).
Start with the tape; visit manual backprop when you want to see the chain rule
spelled out.

---

## 🛠 What's Inside

| Category | The "Fancy" Stuff | Why You Should Care |
| :--- | :--- | :--- |
| **Autograd** | `Tensor`, `GradientTape`, `Function` | A complete reverse-mode AD engine in NumPy. |
| **Attention** | `MHA`, `MQA`, `GQA`, `MLA`, `FlashAttention` | More attention variants than a distracted toddler. `FlashAttention` is API-compatible but simplified — see [status](docs/IMPLEMENTATION_STATUS.md). |
| **Tokenization** | `BPETokenizer`, `RegexTokenizer` | Byte-level BPE with regex splitting. |
| **Vision** | `AlexNet`, `VGG16`, `VGG19`, `im2col` | Classical and modern vision architectures, vectorized. |
| **LLMs** | `LlamaTiny`, `Qwen2Tiny`, `DeepSeekV2Tiny` (MoE) | Tiny models you can actually run on a CPU. |
| **Modern Ops** | `RMSNorm`, `SiLU`, `SwiGLU`, `RoPE` | The secret sauce of modern LLMs, hand-implemented. |
| **Optimizers** | `AdamW`, `Adam`, `SGD+Momentum` | Keep your weights from exploding like a bad science project. |
| **Fine-tuning** | `LoRA`, `QLoRA` | Adapt a model without training every parameter. |

---

## 💻 The Keras You Know

```python
from neutro.models import Sequential
from neutro.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Softmax
from neutro.optimizers import Adam

model = Sequential([
    Conv2D(32, kernel_size=3, input_shape=(28, 28, 1)),
    ReLU(),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10),
    Softmax(),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),   # or optimizer="adam"
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# train_x: (N, 28, 28, 1) float images
# train_y: (N, 10) one-hot labels
model.fit(train_x, train_y, epochs=10, batch_size=32)
```

`compile` accepts both Keras-style names (`"adam"`, `"sgd"`, `"adamw"`) and
optimizer instances.

---

## 📚 Documentation

Start with [Start Here](docs/START_HERE.md), then follow the guided path above.
The full index, with research-paper links, is [docs/README.md](docs/README.md).

Deep dives:

- [Attention mechanisms](docs/layers/attention/mha.md) — MHA, GQA, MQA, MLA, KV cache
- [Convolutional magic](docs/layers/convolutional/conv2d.md) — the `im2col` deep dive
- [Autograd engine](docs/autograd/autograd.md) — `Tensor`, `GradientTape`, `Function`
- [Optimizers](docs/optimizers/optimizers.md) — SGD, Adam, AdamW, schedules
- [Layer lifecycle](docs/layers/base.md) — `build`, `forward`, `backward`

---

## 🧪 Examples

`examples/00_hello_neutro.py` is the first stop. The rest are ordered by
difficulty in [examples/README.md](examples/README.md):

- `mnist_cnn.py` — standard digit classification with a small subset.
- `mnist_functional_residual.py` — Functional API and skip connections.
- `sentiment_gru.py` — embeddings, GRU, padded sequences.
- `reasoning_train.py` — a tiny causal transformer on arithmetic.
- `wikitext_llm.py` — a character-level transformer language model.
- `lora_finetune.py`, `qlora_finetune.py`, `bitnet_train.py` — modern fine-tuning.
- `qwen_moe_toy.py`, `tinystories_diffusion.py`, `mnist_vae.py`, `mnist_diffusion.py` — research toys.

Dataset examples cache downloads in `~/.neutro/datasets`.

---

## 🏗 Installation Options

From a clone (recommended for reading and editing):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

From `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🧭 Status and Scope

Neutro is educational software. It is intentionally naive, likely slow compared
to compiled kernels, and not intended for production. Some components are
simplified or API-compatible aliases; see
[Implementation status](docs/IMPLEMENTATION_STATUS.md) before assuming a name
means the full published algorithm.

If you're looking to change the world with AGI, go to PyTorch. If you're looking
to understand why your Transformer works while drinking a nice cup of tea,
you're in the right place.

---

## 🤝 Contributing

Contributions are welcome, especially documentation and small, focused
examples. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the setup, test command,
and the "add a layer" checklist.

---

## 📄 License

MIT. See [LICENSE](LICENSE).
