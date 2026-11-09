# Tutorial 02 — Gradients and the Tape

**Time:** about 12 minutes. **Prerequisite:** [Tutorial 01](01-first-model.md).

Tutorial 01 showed *that* the loss goes down. This lesson shows *how* the
gradient is computed, and why most Neutro layers only need a `forward` method.

---

## 1. One training step, end to end

```text
x_batch, y_batch
      |
      v
model.forward(x_batch) # predictions
      |
      v
loss_fn(y_batch, predictions) # one scalar
      |
      v
model.backward(dL/dpred) # fill layer.grads
      |
      v
optimizer.step(layers) # change params
```

`loss_fn.gradient(...)` gives the gradient of the scalar loss with respect to
the predictions. `model.backward` propagates that gradient backward through
the layers.

---

## 2. The tape in one example

The autograd engine is small enough to use directly:

```python
import numpy as np
from neutro.autograd import GradientTape, Tensor

x = Tensor(np.array([[1.0, 2.0],
                     [3.0, 4.0]]))
W = Tensor(np.array([[0.5],
                     [-0.5]]))

with GradientTape as tape:
    tape.watch(W)
    loss = ((x @ W) ** 2).mean

tape.gradient(loss, [W])
print(float(loss.data))
print(W.grad)
```

Expected output:

```text
0.25
[[-2.]
 [-3.]]
```

What happened:

1. `tape.watch(W)` says "record operations involving `W`."
2. `x @ W` records a matrix-multiply op with a backward rule.
3. `** 2` records a power op.
4. `.mean` records a reduction.
5. `tape.gradient(loss, [W])` walks the recorded ops in reverse and fills
   `W.grad`.

The tape does **not** keep a graph forever. It records the operations that ran
inside the `with` block, then `gradient` consumes them.

---

## 3. Why `Dense` has no `backward` method

Open `neutro/layers/core/dense.py`. The whole computational part is:

```python
def forward(self, inputs, training=False):
    z = inputs @ self.params["W"]
    if self.use_bias:
        z = z + self.params["b"]
    if self.activation:
        return self.activation(z)
    return z
```

There is no derivative code. So where do `W.grad` and `b.grad` come from?

They come from `Layer.backward` in `neutro/layers/base.py`. In plain English:

1. Take the inputs from the last forward call.
2. Build a `GradientTape`.
3. Watch every parameter in the layer tree.
4. Re-run `forward` inside the tape.
5. Build a scalar probe: `(output * grad_output).sum`.
6. Ask the tape for `d probe / d params`.
7. Copy those gradients into `layer.grads` and return the input gradient.

That is why this pattern is called a **tape-backed fallback**. It works for
any layer whose `forward` is written with `Tensor` operations.

> The probe trick: `(output * g).sum` has the same gradient with respect to
> a parameter as the original scalar loss would, because `g` is exactly the
> upstream gradient. Multiplying and summing turns the vector-Jacobian product
> into an ordinary scalar differentiation problem for the tape.

---

## 4. When a layer writes `backward` by hand

A few layers override `backward` when the explicit chain rule is the lesson,
or when replaying forward is awkward:

- `BitLinear`
- `LoRADense`, `QLoRADense`
- `UNet`
- `VAE`
- `MaskedDiffusionLM`

You can see the list and the reason in
[Implementation status](../IMPLEMENTATION_STATUS.md#backward-strategies).
You do **not** need to read those on the beginner path.

If you do want to write one, the shape rules are:

```text
y = x @ W
dL/dW = x.T @ dL/dy
dL/db = sum(dL/dy, axis=0)
dL/dx = dL/dy @ W.T
```

For a batch of rank-3 inputs, flatten the leading dimensions first:

```text
x: (B, S, D) -> (B*S, D)
dy: (B, S, U) -> (B*S, U)
dW = x_flat.T @ dy_flat # (D, U)
```

---

## 5. `fit` vs `autograd_fit`

Both train a model, but they differ in where the tape is applied:

- `model.fit(...)` — the default. Each layer's `backward` uses the
  tape-backed fallback. This keeps layer code short and is the beginner path.
- `model.autograd_fit(...)` — one tape wraps the whole forward pass and the
  loss. Use this when you want to see the entire computation as one graph.

They should produce the same kind of result. Start with `fit`; read
`Model._train_on_batch` in `neutro/models/base_model.py` when you want the
difference in code.

---

## 6. Check your understanding

1. Why does `tape.watch(W)` matter? What happens if you omit it?
2. In `(x @ W) ** 2`, which operation's backward rule multiplies by `2 * (x @ W)`?
3. Why does `Dense` not need to cache `self.inputs` for its own `backward`?
4. If `x` is `(32, 128)` and `W` is `(128, 64)`, what is `dL/dW`'s shape?

<details>
<summary>Answers</summary>

1. The tape only records ops whose inputs are watched. Without watching, the
   output is not connected to `W` in the tape's recorded graph and `W.grad`
   stays `None`.
2. The power op `** 2`.
3. The base `Layer.backward` re-runs `forward` with the captured inputs, so it
   does not need a cached input from the original call. (Manual-backward layers
   do cache what they need.)
4. `(128, 64)`, the same shape as `W`.

</details>

---

## What to read next

- [Layer lifecycle](../layers/base.md)
- [Autograd engine](../autograd/autograd.md)
- [`Tensor`](../autograd/tensor.md)
- [`GradientTape`](../autograd/tape.md)
- [Optimizers](../optimizers/optimizers.md)
