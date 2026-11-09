# Loss Functions

## Theory

A loss function $L(y_{\text{true}}, y_{\text{pred}})$ measures how wrong the
predictions are. Training minimizes it by gradient descent.

Every loss in Neutro implements:

- `__call__(y_true, y_pred)` → scalar loss.
- `gradient(y_true, y_pred)` → $\partial L / \partial y_{\text{pred}}$, an array
  with the same shape as `y_pred`.

Source: `neutro/losses/`.

---

## Mean Squared Error — `neutro/losses/mse.py`

$$L = \frac{1}{N}\sum_{i=1}^{N}(y_{\text{pred}} - y_{\text{true}})^2$$

$$\frac{\partial L}{\partial y_{\text{pred}}} = \frac{2}{N}(y_{\text{pred}} - y_{\text{true}})$$

`N` is the total number of elements, not the batch size.

---

## Categorical Crossentropy — `neutro/losses/categorical_crossentropy.py`

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c} y_{\text{true},i,c}\log y_{\text{pred},i,c}$$

$$\frac{\partial L}{\partial y_{\text{pred}}} = -\frac{y_{\text{true}}}{y_{\text{pred}} \cdot N}$$

Use with one-hot targets and a `Softmax` output, or use
`SparseCategoricalCrossentropy(from_logits=True)` with raw logits for the
numerically stable path.

---

## Sparse Categorical Crossentropy — `neutro/losses/sparse_categorical_crossentropy.py`

`y_true` contains integer class ids instead of one-hot vectors:

```text
y_true: (B,) or (B, S)
y_pred: (B, V) or (B, S, V)
```

The loss indexes the probability of the true class directly.

- `from_logits=False` (default): `y_pred` is already softmaxed.
- `from_logits=True`: internally applies softmax and uses the combined
  `softmax + cross-entropy` gradient `p - one_hot(target)`, which is more stable.

---

## Masked Sparse Crossentropy — `neutro/losses/masked_sparse_ce.py`

Ignores positions where the mask is zero. Used by masked diffusion language
models. See [masked_sparse_ce.md](masked_sparse_ce.md).

---

## VAE Loss — `neutro/losses/vae_loss.py`

$$L = L_{\text{recon}} + \beta L_{\text{KL}}$$

Combines a reconstruction term with a KL-divergence term that regularizes the
latent distribution. `VAELoss` is instantiated with the model because the KL
term needs access to the encoder's mean/log-variance.

---

## Usage

```python
import numpy as np
from neutro.losses import CategoricalCrossentropy

loss_fn = CategoricalCrossentropy()

y_true = np.array([[0, 1, 0]])
y_pred = np.array([[0.1, 0.8, 0.1]])

print(loss_fn(y_true, y_pred)) # scalar
print(loss_fn.gradient(y_true, y_pred).shape) # (1, 3)
```

`compile` resolves loss names:

```python
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.compile(optimizer="adam", loss=loss_fn) # or an instance
```

---

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). **Deep Learning**. Chapter 6: Loss Functions.
- Kingma, D. P., & Welling, M. (2014). **Auto-Encoding Variational Bayes**. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
