# Pooling Layers

## Theory

Pooling layers reduce the spatial dimensions of feature maps, providing downsampling and local translation invariance.

### MaxPooling2D — `neutro/layers/pooling/maxpooling2d.py`

Slides a window over the input and takes the maximum value in each window:

$$y_{i,j,k} = \max_{p=1..P, q=1..Q} x_{i \cdot s + p,\; j \cdot s + q,\; k}$$

- **Forward**: `np.max` over sliding windows.
- **Backward**: Routes gradient to the position that was the maximum (argmax routing).

### Global Pooling — `neutro/layers/pooling/global_pooling.py`

Reduces each feature map to a single value:

- **GlobalAveragePooling2D**: $y_k = \frac{1}{H \cdot W} \sum_{i,j} x_{i,j,k}$
- **GlobalMaxPooling2D**: $y_k = \max_{i,j} x_{i,j,k}$

Used before the final Dense layer in CNNs to replace Flatten (fewer parameters, no overfitting).

### UpSampling2D — `neutro/layers/pooling/upsampling2d.py$

Increases spatial dimensions by repeating rows and columns (nearest-neighbor upsampling):

$$y_{i \cdot f + p,\; j \cdot f + q,\; k} = x_{i,j,k}$$

- Used in decoder architectures (UNet, GANs).
- Backward: sums the gradient back into the original positions.

## Implementation Guide

All pooling layers are in `neutro/layers/pooling/`. MaxPooling2D uses `im2col` from `conv_utils.py` to unroll windows, then applies `np.max` and `np.argmax` for efficient forward/backward.

```python
# MaxPooling2D key pattern
cols = im2col(x, self.pool_size, self.strides, padding='valid')
max_idx = np.argmax(cols, axis=0)
output = cols[max_idx, np.arange(cols.shape[1])]
# Reshape to output spatial dimensions
```

## Usage Example

```python
from neutro.layers import MaxPooling2D, GlobalAveragePooling2D

pool = MaxPooling2D(pool_size=(2, 2))
x = np.random.randn(2, 28, 28, 16)
y = pool(x)  # shape (2, 14, 14, 16)

gap = GlobalAveragePooling2D()
z = gap(y)   # shape (2, 16)
```

## References

- Springenberg, J. T., et al. (2014). **Striving for Simplicity: The All Convolutional Net**. [arXiv:1412.6806](https://arxiv.org/abs/1412.6806)
- Lin, M., Chen, Q., & Yan, S. (2013). **Network In Network**. [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
