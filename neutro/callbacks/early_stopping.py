from .base import MonitorCallback


class EarlyStopping(MonitorCallback):
    """Stop training when a monitored metric has stopped improving."""

    def __init__(self, monitor='val_loss', patience=0, mode='auto'):
        super().__init__(monitor=monitor, mode=mode)
        self.patience = patience
        self.wait = 0
        self.best = self._init_best()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self._is_improvement(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"Epoch {epoch + 1}: early stopping")
