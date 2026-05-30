# Metrics

## Theory

Metrics quantify model performance during training and evaluation. Unlike losses, metrics are not used for gradient computation — they are only reported for monitoring.

Every metric in `neutro` implements:
- `__call__(y_true, y_pred) → scalar`: compute the metric value.

## Implementation Guide

### File: `neutro/metrics/base.py`

### Accuracy — `neutro/metrics/accuracy.py`

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\arg\max y_{\text{pred},i} = \arg\max y_{\text{true},i})$$

Works with one-hot targets.

### Sparse Accuracy — `neutro/metrics/sparse_accuracy.py`

Same as Accuracy but `y_true` is integer-encoded.

### Precision — `neutro/metrics/precision.py`

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall — `neutro/metrics/recall.py`

$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1 Score — `neutro/metrics/f1_score.py`

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

## Usage Example

```python
from neutro.metrics import Accuracy

acc = Accuracy()
y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
acc_value = acc(y_true, y_pred)  # scalar

# In model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## References

- Keras Metrics API. [Keras.io](https://keras.io/api/metrics/)
