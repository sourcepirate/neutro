# Loss Functions

## Theory

A loss function $L(y_{\text{true}}, y_{\text{pred}})$ measures the discrepancy between predicted and target values. Training minimizes this loss via gradient descent. Every loss in `neutro` implements two methods:

- `forward(y_true, y_pred) → scalar`: compute the loss value.
- `gradient(y_true, y_pred) → ndarray`: compute $\partial L / \partial y_{\text{pred}}$, the gradient w.r.t. the prediction.

## Implementation Guide

### File: `neutro/losses/base.py`

```python
class Loss:
    def forward(self, y_true, y_pred): raise NotImplementedError
    def gradient(self, y_true, y_pred): raise NotImplementedError
```

### Mean Squared Error — `neutro/losses/mse.py`

$$L = \frac{1}{N} \sum_{i=1}^N (y_{\text{pred}} - y_{\text{true}})^2$$

$$\frac{\partial L}{\partial y_{\text{pred}}} = \frac{2}{N} (y_{\text{pred}} - y_{\text{true}})$$

### Categorical Crossentropy — `neutro/losses/categorical_crossentropy.py`

$$L = -\sum_i y_{\text{true},i} \log(y_{\text{pred},i})$$

$$\frac{\partial L}{\partial y_{\text{pred}}} = -\frac{y_{\text{true}}}{y_{\text{pred}}}$$

Used with one-hot encoded targets and Softmax output.

### Sparse Categorical Crossentropy — `neutro/losses/sparse_categorical_crossentropy.py`

Same as categorical crossentropy but `y_true` is integer-encoded (shape `(batch,)`). The loss converts integers to one-hot internally.

### VAE Loss — `neutro/losses/vae_loss.py`

$$L = L_{\text{recon}} + \beta \cdot L_{\text{KL}}$$

Combines a reconstruction loss (e.g., MSE or binary crossentropy) with a KL divergence term that regularizes the latent space.

## Usage Example

```python
from neutro.losses import CategoricalCrossentropy

loss_fn = CategoricalCrossentropy()
y_true = np.array([[0, 1, 0]])
y_pred = np.array([[0.1, 0.8, 0.1]])
l = loss_fn(y_true, y_pred)       # scalar
grad = loss_fn.gradient(y_true, y_pred)  # same shape as y_pred
```

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). **Deep Learning**. Chapter 6: Loss Functions.
