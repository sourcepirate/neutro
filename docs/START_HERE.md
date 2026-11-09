# Start Here

Welcome to **Neutro** — a NumPy-only deep learning library built to be read,
not just run. This page gets you from a fresh clone to a trained model in
about five minutes.

> **Who this is for:** you know Python functions, classes, and a little NumPy,
> but you are new to deep learning. You do not need a GPU, TensorFlow, or
> PyTorch.

---

## 1. Install

```bash
git clone https://github.com/sourcepirate/neutro.git
cd neutro
python -m venv.venv
source.venv/bin/activate # Windows:.venv\Scripts\activate
pip install -e ".[test]"
```

`pip install -e ".[test]"` installs Neutro in editable mode plus `pytest`.
The core library needs `numpy`, `scipy`, `joblib`, `tqdm`, and `regex`;
the examples additionally use `matplotlib` for a couple of plots.

---

## 2. Run your first model

```bash
python examples/00_hello_neutro.py
```

You should see a short summary and a loss that goes down:

```text
Input shape: (256, 8) (batch, features)
Target shape: (256, 3) (batch, classes)

Model summary
-------------------------------------------------------------------------------------
Layer (type) Output Shape Param #
=====================================================================================
hidden (Dense) (None, 16) 144
scores (Dense) (None, 3) 51
=====================================================================================
Total params: 195
Trainable params: 195
Non-trainable params: 0
-------------------------------------------------------------------------------------

Loss after epoch 1: 1.0617
Loss after epoch 5: 0.3720
```

The exact numbers vary slightly with the random seed, but the loss should
fall. That is a complete training loop: forward pass, loss, backward pass,
optimizer step.

### If it fails

- **`ModuleNotFoundError: No module named 'numpy'`** — the virtual environment
  is not active. Run `source.venv/bin/activate` (Windows:
  `.venv\Scripts\activate`) and try again.
- **A dataset download hangs** — the first MNIST/WikiText/IMDB example downloads
  data to `~/.neutro/datasets`. `00_hello_neutro.py` uses synthetic data and
  needs no network.
- **`python` is not found** — use `python3`, or activate the venv first.

---

## 3. Run the tests

```bash
pytest -q
```

All tests should pass. They are also a good source of small, focused examples.

---

## 4. Read in this order

| Step | Read | You will learn |
|------|------|----------------|
| 1 | [Tutorial 01: Your first model](tutorials/01-first-model.md) | `Dense`, shapes, `fit`, and what a gradient looks like |
| 2 | [Tutorial 02: Gradients and the tape](tutorials/02-gradients-and-the-tape.md) | How one training step works end to end |
| 3 | [Tutorial 03: Attention from scratch](tutorials/03-attention-from-scratch.md) | Q/K/V, softmax, heads, causal masks |
| 4 | [Tutorial 04: A tiny transformer LM](tutorials/04-tiny-transformer-lm.md) | Embeddings, causal blocks, training, generation |
| 5 | [Tutorial 05: Manual backprop](tutorials/05-manual-backprop.md) | The chain rule by hand, compared with the tape |
| 6 | [Tutorial 06: Tokenization](tutorials/06-tokenization.md) | Bytes, BPE, special tokens, save/load |
| 7 | [Tutorial 07: LoRA fine-tuning](tutorials/07-lora-finetuning.md) | Freeze the base, train a low-rank update |
| 8 | [Architecture map](ARCHITECTURE.md) | Where every subsystem lives and how a call flows |
| 9 | [Layer lifecycle](layers/base.md) | `build`, `forward`, and the tape-backed `backward` |
| 10 | [Dense layer](layers/core/dense.md) | The linear layer, with shapes and gradients |
| 11 | [Autograd engine](autograd/autograd.md) | How `Tensor` and `GradientTape` work |
| 12 | [API conventions](API_CONVENTIONS.md) | Parameter names, attributes, shapes |
| 13 | [Implementation status](IMPLEMENTATION_STATUS.md) | What is real, simplified, or an alias |

The full documentation index is [docs/README.md](README.md).

---

## The 30-second mental model

1. A **model** is a stack of **layers**.
2. A layer owns **parameters** (arrays) and turns an input into an output in
   `forward`.
3. The model computes a **loss** comparing predictions to targets.
4. The **backward** pass fills `layer.grads` with the derivative of the loss
   with respect to each parameter.
5. The **optimizer** nudges each parameter against its gradient.
6. Repeat over batches and epochs.

In Neutro, most layers only write `forward`. The base `Layer.backward` re-runs
that forward pass inside a `GradientTape` and asks the tape for gradients. A few
advanced layers write `backward` by hand; see
[Implementation status](IMPLEMENTATION_STATUS.md#backward-strategies).

---

## Where to get help

- Open an issue: <https://github.com/sourcepirate/neutro/issues>
- Contribution guide: [CONTRIBUTING.md](https://github.com/sourcepirate/neutro/blob/main/CONTRIBUTING.md)
- Glossary: [GLOSSARY.md](GLOSSARY.md)
