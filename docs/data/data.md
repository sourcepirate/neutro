# Data

## `DataLoader` — `neutro/data.py`

Batches and shuffles arrays. It assumes `x` and `y` support NumPy-style
indexing (`x[indices]`).

```python
from neutro.data import DataLoader

loader = DataLoader(x, y, batch_size=32, shuffle=True)

print(len(loader)) # number of batches
bx, by = loader[0] # one batch

for bx, by in loader:... # reshuffles after the epoch
```

### What each method does

- `__len__` — `ceil(len(x) / batch_size)`.
- `on_epoch_end` — reshuffles `self.indices` when `shuffle=True`.
- `__getitem__(index)` — slices the index array, indexes `x` and `y`, and
  applies the optional `augmenter` to each `x` item.
- `__iter__` — yields every batch, then calls `on_epoch_end`.

### Augmentation

`augmenter` is any object with `apply_transform(x)`. `ImageDataGenerator`
provides one; see [preprocessing](../preprocessing/preprocessing.md).

```python
from neutro.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1 / 255.0)
flow = datagen.flow(x_train, y_train, batch_size=64)
```

### A note on data types

The loader does not convert labels or one-hot encode them. Do that before
constructing it:

```python
import numpy as np

y_one_hot = np.eye(10)[labels]
```

For generators, `Model.fit` accepts an object that yields `(x_batch, y_batch)`
and has `__len__`; `DataLoader` satisfies that contract.
