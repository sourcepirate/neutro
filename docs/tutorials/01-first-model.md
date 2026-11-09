# Tutorial 01 — Your First Model

**Time:** about 10 minutes. **Prerequisites:** [Start Here](../START_HERE.md).

In this lesson we build a tiny classifier, train it, and look at the actual
numbers inside a layer. We will not hide the shapes or the gradient step.

By the end you will be able to answer:

- What does a `Dense` layer compute?
- What shape is the input, the output, and the weight matrix?
- What is `layer.grads['W']`?
- What changed when the loss went down?

---

## 1. The problem

We have 256 examples. Each example is a vector of 8 numbers. Each example
belongs to one of 3 classes.

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(256, 8)) # (samples, features)

scores = x @ rng.normal(size=(8, 3)) # a hidden linear rule
labels = np.argmax(scores, axis=1) # 0, 1, or 2

y = np.eye(3)[labels] # one-hot: (256, 3)
print(x.shape, y.shape)
```

Expected output:

```text
(256, 8) (256, 3)
```

`y` is **one-hot**: `[1, 0, 0]` means class 0, `[0, 1, 0]` means class 1,
and so on. This is the target format expected by categorical crossentropy.

---

## 2. The model

```python
from neutro.layers import Dense
from neutro.models import Sequential
from neutro.optimizers import Adam

model = Sequential([
    Dense(16, activation="relu", input_shape=(8,), name="hidden"),
    Dense(3, activation="softmax", name="scores"),
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary
```

Expected output (the exact spacing is not important):

```text
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
```

Where do the parameter counts come from?

- `hidden`: `W` is `(8, 16)` and `b` is `(16,)` → `8*16 + 16 = 144`.
- `scores`: `W` is `(16, 3)` and `b` is `(3,)` → `16*3 + 3 = 51`.

`None` in the output shape is the batch dimension. It is unknown until you
pass real data.

---

## 3. What `Dense` actually does

The forward pass is:

```text
z = x @ W + b
y = activation(z)
```

With shapes:

```text
x: (B, D) B = batch, D = features
W: (D, U) U = units
b: (U,)
z: (B, U)
y: (B, U)
```

For the first layer: `(256, 8) @ (8, 16) + (16,) -> (256, 16)`.
For the second: `(256, 16) @ (16, 3) + (3,) -> (256, 3)`.

You can inspect the parameters:

```python
W = model.layers[0].params["W"]
b = model.layers[0].params["b"]
print(W.shape, b.shape) # (8, 16) (16,)
print(type(W)) # <class 'neutro.autograd.tensor.Tensor'>
```

`W` is a `Tensor`, not a raw NumPy array. It holds `W.data` (the numbers) and
`W.grad` (the gradient, filled during training).

---

## 4. Train

```python
history = model.fit(x, y, epochs=5, batch_size=32, verbose=0)

losses = np.round(history.history["loss"], 4)
accuracies = np.round(history.history["accuracy"], 4)
print("loss: ", losses)
print("accuracy:", accuracies)
```

Expected output, approximately:

```text
loss: [1.0617 0.8128 0.5904 0.4524 0.372 ]
accuracy: [0.64 0.75 0.85 0.91 0.95]
```

The exact numbers depend on the random initialization and the shuffle. The
important part is the direction: loss goes down, accuracy goes up.

What happened inside one epoch:

1. Shuffle the 256 examples.
2. Split into batches of 32 (8 batches).
3. For each batch:
   - `forward` produces `(32, 3)` probabilities.
   - `categorical_crossentropy` compares them to `(32, 3)` one-hot targets.
   - `backward` fills `layer.grads`.
   - `Adam` updates `W` and `b`.

---

## 5. Look at a gradient

After `fit`, the last batch's gradients are still on the layers:

```python
W = model.layers[0].params["W"]
print(W.grad.shape) # (8, 16)
print(np.round(W.grad[:3,:3], 4))
```

One possible output:

```text
(8, 16)
[[-0.0056 0.0045 0.0035]
 [-0.0031 -0.0182 -0.0221]
 [ 0.0155 0.0234 0.0242]]
```

Each number says: *if this weight were slightly larger, how would the loss
change?* A positive gradient means the loss would increase, so the optimizer
moves the weight in the negative direction:

```text
W_new = W_old - learning_rate * Adam_adjusted_grad
```

That is the whole idea of gradient descent. Everything else in this library is
a more elaborate version of that sentence.

> `W.grad` is a NumPy array, not a `Tensor`. The tape writes gradients as raw
> arrays into `Tensor.grad`; `Layer.backward` then copies them into
> `layer.grads["W"]`.

---

## 6. Predict

```python
probs = model.predict(x[:2])
print(type(probs))
print(probs.shape)
print(probs.data)
```

Expected shape:

```text
(2, 3)
```

`predict` returns a `Tensor`. Use `.data` when you want the raw NumPy array.
Each row has three probabilities that sum to 1 because the final layer uses
`softmax`.

---

## 7. Your turn

Try one change at a time and predict what will happen before you run it.

1. **Remove the hidden layer.** Does the model still learn? Why?
2. **Change 16 units to 2.** Does accuracy get worse? Why might a 2-unit hidden
   layer struggle with 3 classes?
3. **Change the activation to `"tanh"`.** Does training still converge?
4. **Set the learning rate to 1.0.** What happens to the loss? Why?
5. **Print `model.layers[0].params["W"].grad` before and after `fit`.** What is
   it before training?

<details>
<summary>Hint for question 1</summary>

Without a hidden layer, the model is a single linear map followed by softmax.
It can still separate classes that are linearly separable, but it cannot bend
the decision boundary. That is why nonlinear activations matter.

</details>

---

## What to read next

- [Tutorial 02 — Gradients and the tape](02-gradients-and-the-tape.md)
- [Layer lifecycle](../layers/base.md)
- [Dense layer walkthrough](../layers/core/dense.md)
- [Architecture map](../ARCHITECTURE.md)
