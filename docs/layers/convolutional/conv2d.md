# Conv2D and the im2col Algorithm

## Overview
Convolutional layers are the building blocks of CNNs. To achieve high performance in NumPy without specialized CUDA kernels, we use the `im2col` (image to column) transformation.

## Algorithm: im2col
The `im2col` algorithm transforms a 4D input volume (Batch, Height, Width, Channels) into a 2D matrix where each column represents a receptive field (patch) from the input.

1.  **Padding**: Pad the input volume if `padding='same'`.
2.  **Extraction**: Slide the filter window across the input and "unroll" each 3D patch (Kernel Height $\times$ Kernel Width $\times$ Input Channels) into a single column.
3.  **Matrix Multiplication**: The convolution becomes a single large matrix multiplication:
    $$\text{Output} = W_{\text{flat}} \times X_{\text{col}} + b$$
4.  **Reshape**: Reshape the 2D result back to the 4D output volume.

## Implementation Details
We use fancy indexing in NumPy to implement `im2col` efficiently in `neutro/utils/conv_utils.py`. The `col2im` operation is used during backpropagation to accumulate gradients back into the input volume shape.

## References
- CS231n: Convolutional Neural Networks for Visual Recognition. **Convolutional Layers**. [Stanford University](https://cs231n.github.io/convolutional-networks/#conv).
- High Performance Hardware for Machine Learning. **The im2col transformation**.
