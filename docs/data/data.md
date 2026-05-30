# Data

## DataLoader — `neutro/data.py`

A simple data loader for batching and shuffling:

```python
class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield self.dataset[batch_idx]
```
