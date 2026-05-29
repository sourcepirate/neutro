import numpy as np
import pytest
from neutro.callbacks import ReduceLROnPlateau
from neutro.models import Sequential
from neutro.layers import Dense
from neutro.optimizers import SGD

def test_reduce_lr_on_plateau():
    model = Sequential([Dense(1, input_shape=(1,))])
    optimizer = SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='mse')
    
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=0.01)
    reduce_lr.set_model(model)
    reduce_lr.on_train_begin()
    
    # Epoch 1: loss 1.0
    reduce_lr.on_epoch_end(0, logs={'loss': 1.0})
    assert model.optimizer.lr == 0.1
    
    # Epoch 2: loss 1.0 (no improvement)
    reduce_lr.on_epoch_end(1, logs={'loss': 1.0})
    assert model.optimizer.lr == 0.1
    
    # Epoch 3: loss 1.0 (no improvement, wait=2 >= patience=2)
    reduce_lr.on_epoch_end(2, logs={'loss': 1.0})
    assert model.optimizer.lr == 0.05
    
    # Epoch 4: loss 0.5 (improvement)
    reduce_lr.on_epoch_end(3, logs={'loss': 0.5})
    assert model.optimizer.lr == 0.05
    assert reduce_lr.wait == 0
