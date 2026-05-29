import numpy as np

class DataLoader:
    """
    Data loader for batching and shuffling data.
    
    Args:
        x: Input data (NumPy array).
        y: Target data (NumPy array).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at the beginning of each epoch.
    """
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
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_x, batch_y = self.x[indices], self.y[indices]
        
        if self.augmenter:
            augmented_x = np.zeros_like(batch_x)
            for i in range(len(batch_x)):
                augmented_x[i] = self.augmenter.apply_transform(batch_x[i])
            batch_x = augmented_x
            
        return batch_x, batch_y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        self.on_epoch_end()
