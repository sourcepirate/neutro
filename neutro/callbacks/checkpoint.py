import numpy as np
from .base import Callback

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', save_best_only=False, mode='auto'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = -np.inf if mode == 'max' or (mode == 'auto' and 'acc' in monitor) else np.inf
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None: return

        if self.save_best_only:
            if (self.mode == 'min' and current < self.best) or \
               (self.mode == 'max' and current > self.best) or \
               (self.mode == 'auto' and (('acc' in self.monitor and current > self.best) or ('loss' in self.monitor and current < self.best))):
                self.best = current
                self.model.save(self.filepath)
        else:
            self.model.save(self.filepath.format(epoch=epoch + 1, **logs))
