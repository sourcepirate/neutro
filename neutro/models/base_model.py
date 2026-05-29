import numpy as np
import joblib
from .. import metrics as metrics_module
from .. import losses as losses_module
from ..callbacks import History

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

    def _get_all_layers(self, layers=None):
        if layers is None:
            layers = self.layers
        
        all_layers = []
        for layer in layers:
            all_layers.append(layer)
            if hasattr(layer, 'sublayers'):
                all_layers.extend(self._get_all_layers(layer.sublayers))
        return all_layers

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
                all_trainable_layers = self._get_all_layers()
                self.optimizer.step(all_trainable_layers)
                
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

    def forward(self, inputs, training=False, kv_cache=None):
        for i, layer in enumerate(self.layers):
            if kv_cache is not None and hasattr(layer, 'forward'):
                # Check if layer accepts kv_cache (Attention or Blocks)
                import inspect
                sig = inspect.signature(layer.forward)
                if 'kv_cache' in sig.parameters:
                    inputs = layer(inputs, training=training, kv_cache=kv_cache, layer_id=i)
                else:
                    inputs = layer(inputs, training=training)
            else:
                inputs = layer(inputs, training=training)
        return inputs

    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        """
        Autoregressive generation with KV Caching.
        """
        from ..layers.attention.kv_cache import KVCache
        cache = KVCache()
        
        # Start with the full prompt
        curr_tokens = start_tokens
        generated = start_tokens
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(curr_tokens, training=False, kv_cache=cache)
            
            # Get the logits for the last token only: (batch, seq, vocab) -> (batch, vocab)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Simple sample or argmax
            # For "naive" let's do argmax or simple categorical sample
            probs = np.exp(next_token_logits - np.max(next_token_logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            
            # Sample next token
            next_token = np.array([np.random.choice(len(p), p=p) for p in probs])
            next_token = next_token.reshape(-1, 1)
            
            # Update inputs for next step
            # With KV Cache, we only need to pass the LAST token
            curr_tokens = next_token
            generated = np.concatenate([generated, next_token], axis=1)
            
        return generated

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

    def summary(self):
        """
        Prints a Keras-style summary of the model.
        """
        print("-" * 65)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 65)
        
        total_params = 0
        trainable_params = 0
        
        # We need an initial input shape. If not built, we might not know.
        # Sequential models usually have input_shape in the first layer.
        curr_shape = None
        if self.layers and self.layers[0].input_shape:
            curr_shape = self.layers[0].input_shape

        for layer in self.layers:
            name = layer.name or layer.__class__.__name__
            layer_type = layer.__class__.__name__
            
            if curr_shape is not None:
                try:
                    output_shape = layer.compute_output_shape(curr_shape)
                    curr_shape = output_shape
                except Exception:
                    output_shape = "multiple"
            else:
                output_shape = "unbuilt"
            
            params = layer.count_params()
            total_params += params
            if getattr(layer, 'trainable', True):
                trainable_params += params
            
            print(f"{name + ' (' + layer_type + ')':<25} {str(output_shape):<20} {params:<10,}")
            
        print("=" * 65)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("-" * 65)

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
        if not self.layers:
            # If it's the first layer and has input_shape, build it
            if layer.input_shape:
                shape = layer.input_shape
                # If input_shape doesn't include batch, add it
                # Convention: if rank is 1 (seq_len) or 2 (flat) or 3 (image), we might need batch
                # But it's safer to just check if the user provided it.
                # In LlamaTiny, input_shape=(seq_len,) is passed.
                if len(shape) == 1: 
                    shape = (None,) + shape
                elif len(shape) == 3: # (h, w, c)
                    shape = (None,) + shape
                layer.build(shape)
        else:
            # Build based on previous layer's output shape
            prev_layer = self.layers[-1]
            if prev_layer.built:
                input_shape = prev_layer.compute_output_shape(prev_layer.input_shape)
                layer.build(input_shape)
        
        self.layers.append(layer)
