import numpy as np
import os
from neutro.models import Sequential
from neutro.layers.core.dense import Dense
from neutro.optimizers.sgd import SGD
from neutro.losses.mse import MeanSquaredError
from neutro.callbacks import EarlyStopping, ModelCheckpoint

def test_early_stopping():
    model = Sequential([Dense(1)])
    model.compile(optimizer=SGD(), loss=MeanSquaredError())
    x = np.random.rand(10, 1)
    y = np.random.rand(10, 1)
    # Loss will likely not improve much, so it might stop
    es = EarlyStopping(monitor='loss', patience=0)
    model.fit(x, y, epochs=10, callbacks=[es], verbose=0)
    # Just checking it runs without error

def test_checkpoint(tmp_path):
    model = Sequential([Dense(1)])
    model.compile(optimizer=SGD(), loss=MeanSquaredError())
    x = np.random.rand(10, 1)
    y = np.random.rand(10, 1)
    filepath = os.path.join(tmp_path, "best.joblib")
    cp = ModelCheckpoint(filepath, monitor='loss', save_best_only=True)
    model.fit(x, y, epochs=5, callbacks=[cp], verbose=0)
    assert os.path.exists(filepath)
