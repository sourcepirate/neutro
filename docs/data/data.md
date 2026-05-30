# Data

## DataLoader — `neutro/data.py`

A simple data loader for batching and shuffling:

```python
class DataLoader:
    def __init__(self, x, y, batch_size=32, shuffle=True, augmenter=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenter = augmenter
        self.indices = np.arange(len(x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x, batch_y = self.x[batch_idx], self.y[batch_idx]
        return batch_x, batch_y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        self.on_epoch_end()
```
