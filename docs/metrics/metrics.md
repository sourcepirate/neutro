# Metrics

Metrics measure model performance for reporting. Unlike losses, they do not
produce gradients.

Every metric implements `__call__(y_true, y_pred) -> scalar` and
`get_name -> str`.

Source: `neutro/metrics/`.

---

## Accuracy — `neutro/metrics/accuracy.py`

Compares argmax predictions with argmax targets (or rounded values for binary
outputs):

$$\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\arg\max y_{\text{pred},i} = \arg\max y_{\text{true},i})$$

Use with one-hot targets.

---

## Sparse Accuracy — `neutro/metrics/sparse_accuracy.py`

Compares integer targets directly:

```text
y_true: (B,) or (B, S) integers
y_pred: (B, C) or (B, S, C) probabilities/logits
```

---

## Precision, Recall, F1 — `neutro/metrics/precision.py`, `recall.py`, `f1_score.py`

$$\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

These are binary metrics; pass binary predictions/targets.

---

## Usage

```python
import numpy as np
from neutro.metrics import Accuracy, SparseAccuracy

y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])

print(Accuracy()(y_true, y_pred)) # 1.0

print(SparseAccuracy()(
    np.array([1, 0]),
    np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]),
)) # 1.0
```

In a model:

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

For integer labels, use `metrics=["sparse_accuracy"]`.

---

## References

- Keras Metrics API: <https://keras.io/api/metrics/>
