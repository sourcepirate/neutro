import numpy as np
import os
from neutro.models import Sequential
from neutro.layers.core.dense import Dense
from neutro.optimizers.adam import Adam
from neutro.losses.mse import MeanSquaredError

def test_sequential_fit():
    model = Sequential([Dense(10, activation='relu'), Dense(1)])
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['accuracy'])
    x = np.random.rand(20, 5)
    y = np.random.rand(20, 1)
    history = model.fit(x, y, epochs=2, batch_size=5, verbose=0)
    assert len(history.history['loss']) == 2

def test_sequential_fit_with_val():
    model = Sequential([Dense(2)])
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['accuracy'])
    x = np.random.rand(10, 5)
    y = np.random.rand(10, 2)
    val_data = (np.random.rand(5, 5), np.random.rand(5, 2))
    history = model.fit(x, y, epochs=1, batch_size=2, verbose=1, validation_data=val_data)
    assert 'val_loss' in history.history

def test_evaluate():
    model = Sequential([Dense(2)])
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['accuracy'])
    x = np.random.rand(5, 5)
    y = np.random.rand(5, 2)
    model.predict(x)
    results = model.evaluate(x, y)
    assert 'loss' in results
    assert 'accuracy' in results

def test_save_load(tmp_path):
    model = Sequential([Dense(5)])
    filepath = os.path.join(tmp_path, "model.joblib")
    model.save(filepath)
    loaded = Sequential.load(filepath)
    assert len(loaded.layers) == 1
