import numpy as np
from .base import Callback

class LearningRateScheduler(Callback):
    """
    Learning rate scheduler.
    
    Args:
        schedule: a function that takes an epoch index (integer, indexed from 0) and current learning rate as inputs and returns a new learning rate as output (float).
    """
    def __init__(self, schedule, verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        
        lr = float(self.model.optimizer.lr)
        lr = self.schedule(epoch, lr)
        self.model.optimizer.lr = lr
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: LearningRateScheduler setting learning rate to {lr}.')

class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when a metric has stopped improving.
    """
    def __init__(self, monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.wait = 0
        self.best = np.inf if 'loss' in monitor else -np.inf
        self.cooldown_counter = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.inf if 'loss' in self.monitor else -np.inf
        self.cooldown_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0

        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
        elif self.cooldown_counter <= 0:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.model.optimizer.lr)
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self.model.optimizer.lr = new_lr
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.')
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def _is_improvement(self, current, best):
        if 'loss' in self.monitor or self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
