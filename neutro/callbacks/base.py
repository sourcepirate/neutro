from typing import Any, Dict, Optional


class Callback:
    """Base class for training callbacks.

    Subclasses override the lifecycle hooks below to observe or modify
    training. `set_model` is called before training begins.
    """

    def __init__(self) -> None:
        self.model: Optional[Any] = None

    def set_model(self, model) -> None:
        self.model = model

    def on_train_begin(self, logs=None) -> None:
        pass

    def on_train_end(self, logs=None) -> None:
        pass

    def on_epoch_begin(self, epoch, logs=None) -> None:
        pass

    def on_epoch_end(self, epoch, logs=None) -> None:
        pass

    def on_batch_begin(self, batch, logs=None) -> None:
        pass

    def on_batch_end(self, batch, logs=None) -> None:
        pass


class MonitorCallback(Callback):
    """Base class for callbacks that track an epoch-level metric.

    Provides shared logic for deciding whether a monitored metric has
    improved, mirroring the Keras ``monitor``/``mode`` semantics.
    """

    def __init__(self, monitor: str = 'val_loss', mode: str = 'auto') -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode

    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self.best
        if self.mode == 'max':
            return current > self.best
        # auto: infer direction from the monitored metric's name
        if 'acc' in self.monitor:
            return current > self.best
        if 'loss' in self.monitor:
            return current < self.best
        return current < self.best

    def _init_best(self) -> float:
        maximize = self.mode == 'max' or (self.mode == 'auto' and 'acc' in self.monitor)
        return -float('inf') if maximize else float('inf')
