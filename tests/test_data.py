import numpy as np
import pytest
from neutro.data import DataLoader

def test_dataloader_basic():
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 1)
    batch_size = 32
    
    loader = DataLoader(x, y, batch_size=batch_size, shuffle=False)
    
    assert len(loader) == 4 # 32*3=96, so 4 batches
    
    batches = list(loader)
    assert len(batches) == 4
    
    bx, by = batches[0]
    assert bx.shape == (32, 10)
    assert by.shape == (32, 1)
    
    # Last batch
    bx, by = batches[3]
    assert bx.shape == (4, 10)
    assert by.shape == (4, 1)

def test_dataloader_shuffle():
    x = np.arange(100)
    y = np.arange(100)
    loader = DataLoader(x, y, batch_size=10, shuffle=True)
    
    first_epoch_indices = loader.indices.copy()
    loader.on_epoch_end()
    second_epoch_indices = loader.indices.copy()
    
    assert not np.array_equal(first_epoch_indices, second_epoch_indices)
