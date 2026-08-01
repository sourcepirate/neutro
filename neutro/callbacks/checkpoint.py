from .base import MonitorCallback


class ModelCheckpoint(MonitorCallback):
    """Save the model to disk after each epoch (optionally only on improvement)."""

    def __init__(self, filepath, monitor='val_loss', save_best_only=False, mode='auto'):
        super().__init__(monitor=monitor, mode=mode)
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best = self._init_best()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.save_best_only:
            if self._is_improvement(current):
                self.best = current
                self.model.save(self.filepath)
        else:
            self.model.save(self.filepath.format(epoch=epoch + 1, **logs))
