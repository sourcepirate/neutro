# Callbacks

## Theory

Callbacks are objects that hook into the training loop at various points. They allow you to monitor training, save checkpoints, adjust learning rates, and stop training early without cluttering the training loop itself.

**Hook points** (in order):
1. `on_train_begin` / `on_train_end`
2. `on_epoch_begin` / `on_epoch_end`
3. `on_batch_begin` / `on_batch_end`

## Implementation Guide

### File: `neutro/callbacks/base.py`

```python
class Callback:
    def set_model(self, model): ...
    def on_train_begin(self, logs=None): ...
    def on_train_end(self, logs=None): ...
    def on_epoch_begin(self, epoch, logs=None): ...
    def on_epoch_end(self, epoch, logs=None): ...
    def on_batch_begin(self, batch, logs=None): ...
    def on_batch_end(self, batch, logs=None): ...
```

All methods are no-ops by default. Subclasses override the needed hooks.

### History — `neutro/callbacks/history.py`

Records per-epoch metrics into `history.history` dict (keys: `loss`, `val_loss`, `accuracy`, etc.).

### EarlyStopping — `neutro/callbacks/early_stopping.py`

Monitors a metric (e.g., `val_loss`) and stops training if it hasn't improved for `patience` epochs. Uses `model.stop_training = True`.

### ReduceLROnPlateau / LR Scheduler — `neutro/callbacks/lr_scheduler.py`

Reduces the learning rate when a metric plateaus, or follows a predefined schedule.

### Checkpoint — `neutro/callbacks/checkpoint.py`

Saves the model to disk at the end of each epoch using `joblib.dump`.

## Usage Example

```python
from neutro.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint('best_model.pkl', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
]
model.fit(X, y, callbacks=callbacks, epochs=100)
```

## References

- Keras Callbacks API. [Keras.io](https://keras.io/api/callbacks/)
