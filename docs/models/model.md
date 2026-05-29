# Sequential and Model API

## Overview
The `Model` and `Sequential` classes provide the top-level API for building and training neural networks, mimicking the Keras `Model` and `Sequential` classes.

## Key Methods
- **`compile(optimizer, loss, metrics)`**: Configures the model for training.
- **`fit(x, y, epochs, batch_size, ...)`**: Trains the model for a fixed number of epochs.
- **`evaluate(x, y)`**: Returns the loss value and metrics for the model in test mode.
- **`predict(x)`**: Generates output predictions for the input samples.
- **`save(filepath)` / `load(filepath)`**: Serialization using `joblib`.

## Implementation Details
The training loop in `fit` handles batching, shuffling (via `DataLoader`), forward and backward passes, optimizer steps, and callback execution.

## References
- Chollet, F. (2015). **Keras**. [GitHub](https://github.com/keras-team/keras).
- [TensorFlow Documentation: The Sequential model](https://www.tensorflow.org/guide/keras/sequential_model).
