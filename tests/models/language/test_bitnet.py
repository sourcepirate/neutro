import numpy as np
from neutro.models.language.bitnet import BitNetTiny
from neutro.optimizers import AdamW


def test_bitnet_smoke():
    vocab_size = 100
    seq_len = 10
    model = BitNetTiny(vocab_size, seq_len, dim=32, n_layers=1, n_heads=4, mode='b1.58')
    x = np.random.randint(0, vocab_size, (1, seq_len))
    y = model.predict(x)
    assert y.shape == (1, seq_len, vocab_size)


def test_bitnet_b1_smoke():
    vocab_size = 100
    seq_len = 10
    model = BitNetTiny(vocab_size, seq_len, dim=32, n_layers=1, n_heads=4, mode='b1')
    x = np.random.randint(0, vocab_size, (1, seq_len))
    y = model.predict(x)
    assert y.shape == (1, seq_len, vocab_size)


def test_bitnet_train_step():
    vocab_size = 100
    seq_len = 10
    model = BitNetTiny(vocab_size, seq_len, dim=32, n_layers=1, n_heads=4)
    model.compile(optimizer=AdamW(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_accuracy'])
    x = np.random.randint(0, vocab_size, (4, seq_len))
    y = np.random.randint(0, vocab_size, (4, seq_len))
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
    assert 'loss' in history.history
    assert len(history.history['loss']) == 1
