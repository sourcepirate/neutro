import numpy as np
from .base import Callback

class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=0, mode='auto'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.wait = 0
        self.best = -np.inf if mode == 'max' or (mode == 'auto' and 'acc' in monitor) else np.inf
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None: return

        if (self.mode == 'min' and current < self.best) or \
           (self.mode == 'max' and current > self.best) or \
           (self.mode == 'auto' and (('acc' in self.monitor and current > self.best) or ('loss' in self.monitor and current < self.best))):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"Epoch {epoch+1}: early stopping")
