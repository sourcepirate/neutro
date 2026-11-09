# How to Train on MNIST

This guide walks through `examples/mnist_cnn.py`, the smallest realistic image
example. It uses a convolutional network, a small subset of MNIST, and the
`ImageDataGenerator` for rescaling.

## Run it

```bash
python examples/mnist_cnn.py
```

The first run downloads `mnist.npz` to `~/.neutro/datasets`. Later runs reuse
the cached file.

## What the script does

1. Loads MNIST with `load_mnist`.
2. Reshapes images from `(N, 28, 28)` to `(N, 28, 28, 1)` so the channel
   dimension is explicit.
3. One-hot encodes labels with `np.eye(10)[labels]`.
4. Builds a `Sequential` model:

   ```text
   Conv2D(32, 3x3) -> ReLU -> MaxPooling2D
   Conv2D(64, 3x3) -> ReLU -> MaxPooling2D
   Flatten -> Dense(128) -> ReLU -> Dropout(0.5)
   Dense(10) -> Softmax
   ```

5. Compiles with `Adam(0.001)` and categorical crossentropy.
6. Rescales pixels to `[0, 1]` with `ImageDataGenerator(rescale=1/255.0)`.
7. Trains on a 1,000-image subset for 10 epochs.
8. Evaluates on 100 test images.

The subset is intentional: this is a "watch it learn" example, not a benchmark.
Expect modest accuracy, not state of the art.

## Shapes

```text
input: (B, 28, 28, 1)
conv1: (B, 26, 26, 32) valid padding
pool1: (B, 13, 13, 32)
conv2: (B, 11, 11, 64)
pool2: (B, 5, 5, 64)
flatten: (B, 1600)
dense: (B, 128)
output: (B, 10)
```

The exact spatial sizes depend on the padding and stride defaults; run
`model.summary` to see them for your configuration.

## Make it better

- Remove the `[:1000]` subset to train on all 60,000 images.
- Add `ImageDataGenerator(rotation_range=10, horizontal_flip=False)` to
  experiment with augmentation.
- Increase `epochs` and watch the loss curve in `history.history['loss']`.
- Replace `Softmax` + categorical crossentropy with raw logits +
  `SparseCategoricalCrossentropy(from_logits=True)` to see the numerically
  stable version.

## Related

- [Convolution walkthrough](../layers/convolutional/conv2d.md)
- [Pooling walkthrough](../layers/pooling/pooling.md)
- [How to run examples](run-examples.md)
