from .base import Callback

class History(Callback):
    def on_train_begin(self, logs=None):
        self.history = {'loss': [], 'epoch': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history['epoch'].append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
