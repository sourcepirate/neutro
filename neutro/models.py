import numpy as np
import joblib
from . import metrics as metrics_module
from . import losses as losses_module
from .callbacks import History

class Model:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        self.stop_training = False

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss_fn = losses_module.get(loss)
        self.metrics = [metrics_module.get(m) for m in (metrics or [])]

    def fit(self, x, y=None, epochs=1, batch_size=32, verbose=1, validation_data=None, callbacks=None):
        if hasattr(x, '__iter__') and not isinstance(x, np.ndarray):
            use_generator = True
            n_samples = len(x) * x.batch_size if hasattr(x, 'batch_size') else len(x)
        else:
            use_generator = False
            n_samples = x.shape[0]
            
        history = History()
        history.set_model(self)
        
        all_callbacks = [history] + (callbacks or [])
        for cb in all_callbacks:
            cb.set_model(self)
        
        logs = {}
        for cb in all_callbacks: cb.on_train_begin(logs)

        for epoch in range(epochs):
            if self.stop_training: break
            for cb in all_callbacks: cb.on_epoch_begin(epoch, logs)
            
            epoch_loss = 0
            epoch_metrics = {m.get_name(): 0 for m in self.metrics}
            
            if use_generator:
                num_batches = len(x)
                data_iter = iter(x)
            else:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                x_shuffled = x[indices]
                y_shuffled = y[indices]
                num_batches = int(np.ceil(n_samples / batch_size))
            
            total_seen = 0
            for i in range(num_batches):
                if use_generator:
                    x_batch, y_batch = next(data_iter)
                else:
                    start, end = i * batch_size, min((i + 1) * batch_size, n_samples)
                    x_batch = x_shuffled[start:end]
                    y_batch = y_shuffled[start:end]
                
                batch_size_actual = len(x_batch)
                total_seen += batch_size_actual
                
                for cb in all_callbacks: cb.on_batch_begin(i, logs)
                
                # Forward
                output = self.forward(x_batch, training=True)
                
                # Loss & Metrics
                batch_loss = self.loss_fn(y_batch, output)
                epoch_loss += batch_loss * batch_size_actual
                
                for m in self.metrics:
                    epoch_metrics[m.get_name()] += m(y_batch, output) * batch_size_actual
                
                # Backward
                grad = self.loss_fn.gradient(y_batch, output)
                self.backward(grad)
                
                # Update
                self.optimizer.step(self.layers)
                
                for cb in all_callbacks: cb.on_batch_end(i, logs)
            
            logs = {
                'loss': epoch_loss / total_seen,
                **{k: v / total_seen for k, v in epoch_metrics.items()}
            }
            
            if validation_data:
                val_x, val_y = validation_data
                val_output = self.predict(val_x)
                logs['val_loss'] = self.loss_fn(val_y, val_output)
                for m in self.metrics:
                    logs[f'val_{m.get_name()}'] = m(val_y, val_output)

            for cb in all_callbacks: cb.on_epoch_end(epoch, logs)
            
            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f}"
                for m in self.metrics:
                    name = m.get_name()
                    msg += f" - {name}: {logs[name]:.4f}"
                if validation_data:
                    msg += f" - val_loss: {logs['val_loss']:.4f}"
                    for m in self.metrics:
                        name = m.get_name()
                        msg += f" - val_{name}: {logs[f'val_{name}']:.4f}"
                print(msg)
        
        for cb in all_callbacks: cb.on_train_end(logs)
        return history

    def forward(self, inputs, training=False):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x):
        return self.forward(x, training=False)

    def evaluate(self, x, y):
        output = self.predict(x)
        loss = self.loss_fn(y, output)
        results = {'loss': loss}
        for m in self.metrics:
            results[m.get_name()] = m(y, output)
        return results

    def save(self, filepath):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)

class Sequential(Model):
    def __init__(self, layers=None):
        super().__init__()
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self.layers.append(layer)
