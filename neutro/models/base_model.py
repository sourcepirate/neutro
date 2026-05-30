import numpy as np
import joblib
from tqdm import tqdm
from .. import metrics as metrics_module
from .. import losses as losses_module
from ..callbacks import History

from ..layers.base import Layer

class Model(Layer):
    def __init__(self, inputs=None, outputs=None, name=None):
        super().__init__(name=name)
        self.layers = []
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        self.stop_training = False
        
        self.inputs = inputs
        self.outputs = outputs
        
        if inputs is not None and outputs is not None:
            self._init_graph(inputs, outputs)

    def _init_graph(self, inputs, outputs):
        """
        Traverses the graph from outputs to inputs to discover all layers and nodes.
        """
        from ..engine.node import Node
        
        self._nodes_by_depth = []
        self._layers = []
        
        # Topological sort
        visited_nodes = set()
        nodes_ordered = []
        
        def traverse(tensor):
            if hasattr(tensor, 'node') and tensor.node:
                node = tensor.node
                if node not in visited_nodes:
                    visited_nodes.add(node)
                    # Recursive call for all input tensors of this node
                    if isinstance(node.input_tensors, list):
                        for t in node.input_tensors:
                            traverse(t)
                    else:
                        traverse(node.input_tensors)
                    nodes_ordered.append(node)
        
        if isinstance(outputs, list):
            for o in outputs:
                traverse(o)
        else:
            traverse(outputs)
            
        self._nodes_ordered = nodes_ordered
        
        # Collect all unique layers
        for node in nodes_ordered:
            if node.layer not in self.layers:
                self.layers.append(node.layer)

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss_fn = losses_module.get(loss)
        self.metrics = [metrics_module.get(m) for m in (metrics or [])]

    def _get_all_layers(self, layers=None, visited=None):
        if layers is None:
            layers = self.layers
        if visited is None:
            visited = set()
        
        all_layers = []
        for layer in layers:
            l_id = id(layer)
            if l_id not in visited:
                visited.add(l_id)
                all_layers.append(layer)
                if hasattr(layer, 'sublayers'):
                    all_layers.extend(self._get_all_layers(layer.sublayers, visited))
        return all_layers

    _STATE_EXCLUDE = {'params', 'grads', 'built', 'input_shape', 'output_shape',
                      'name', '_inbound_nodes', 'trainable'}

    @staticmethod
    def _capture_layer_state(layer):
        """Recursively capture state of a layer and all its sublayers.
        Returns dict: {id(sublayer): {attr_name: value, ...}}"""
        state = {}
        stack = [layer]
        visited = set()
        while stack:
            l = stack.pop()
            l_id = id(l)
            if l_id in visited:
                continue
            visited.add(l_id)
            sub = {}
            for k, v in l.__dict__.items():
                if k not in Model._STATE_EXCLUDE:
                    sub[k] = v
            state[l_id] = sub
            for sl in l.sublayers:
                stack.append(sl)
        return state

    @staticmethod
    def _restore_layer_state(layer, state):
        """Restore state captured by _capture_layer_state onto layer tree."""
        stack = [layer]
        visited = set()
        while stack:
            l = stack.pop()
            l_id = id(l)
            if l_id in visited:
                continue
            visited.add(l_id)
            if l_id in state:
                for k, v in state[l_id].items():
                    setattr(l, k, v)
            for sl in l.sublayers:
                stack.append(sl)

    def fit(self, x, y=None, epochs=1, batch_size=32, verbose=1, validation_data=None, callbacks=None):
        is_mimo_x = isinstance(x, list)
        is_mimo_y = isinstance(y, list)
        
        use_generator = False
        if not is_mimo_x and hasattr(x, '__iter__') and not isinstance(x, np.ndarray):
            use_generator = True
            n_samples = len(x) * x.batch_size if hasattr(x, 'batch_size') else len(x)
        else:
            if is_mimo_x:
                n_samples = x[0].shape[0]
            else:
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
                
                if is_mimo_x:
                    x_shuffled = [xi[indices] for xi in x]
                else:
                    x_shuffled = x[indices]
                
                if is_mimo_y:
                    y_shuffled = [yi[indices] for yi in y]
                else:
                    y_shuffled = y[indices]
                    
                num_batches = int(np.ceil(n_samples / batch_size))
            
            total_seen = 0
            if verbose == 1:
                pbar = tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}")
            
            for i in range(num_batches):
                if use_generator:
                    x_batch, y_batch = next(data_iter)
                else:
                    start, end = i * batch_size, min((i + 1) * batch_size, n_samples)
                    
                    if is_mimo_x:
                        x_batch = [xi[start:end] for xi in x_shuffled]
                    else:
                        x_batch = x_shuffled[start:end]
                    
                    if is_mimo_y:
                        y_batch = [yi[start:end] for yi in y_shuffled]
                    else:
                        y_batch = y_shuffled[start:end]
                
                batch_size_actual = x_batch[0].shape[0] if is_mimo_x else len(x_batch)
                total_seen += batch_size_actual
                
                for cb in all_callbacks: cb.on_batch_begin(i, logs)
                
                # Forward
                output = self.forward(x_batch, training=True)
                
                # Loss - sum across multiple outputs if applicable
                is_mimo_out = isinstance(self.outputs, list)
                if is_mimo_out:
                    batch_loss = sum(self.loss_fn(y_batch[j], output[j]) for j in range(len(self.outputs)))
                else:
                    batch_loss = self.loss_fn(y_batch, output)
                epoch_loss += batch_loss * batch_size_actual
                
                for m in self.metrics:
                    try:
                        m_val = m(y_batch, output)
                    except (TypeError, ValueError):
                        m_val = m(y_batch[0], output[0]) if is_mimo_out else 0.0
                    epoch_metrics[m.get_name()] += m_val * batch_size_actual
                
                # Backward
                if is_mimo_out:
                    grads = [self.loss_fn.gradient(y_batch[j], output[j]) for j in range(len(self.outputs))]
                    self.backward(grads)
                else:
                    grad = self.loss_fn.gradient(y_batch, output)
                    self.backward(grad)
                
                # Update
                all_trainable_layers = self._get_all_layers()
                self.optimizer.step(all_trainable_layers)
                
                for cb in all_callbacks: cb.on_batch_end(i, logs)

                if verbose == 1:
                    postfix = {'loss': f"{epoch_loss / total_seen:.4f}"}
                    for m in self.metrics:
                        name = m.get_name()
                        postfix[name] = f"{epoch_metrics[name] / total_seen:.4f}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)
            
            if verbose == 1:
                pbar.close()

            logs = {
                'loss': epoch_loss / total_seen,
                **{k: v / total_seen for k, v in epoch_metrics.items()}
            }
            
            if validation_data:
                val_x, val_y = validation_data
                val_output = self.predict(val_x)
                is_mimo_val_out = isinstance(self.outputs, list)
                if is_mimo_val_out:
                    logs['val_loss'] = sum(self.loss_fn(val_y[j], val_output[j]) for j in range(len(self.outputs)))
                else:
                    logs['val_loss'] = self.loss_fn(val_y, val_output)
                for m in self.metrics:
                    try:
                        m_val = m(val_y, val_output)
                    except (TypeError, ValueError):
                        m_val = m(val_y[0], val_output[0]) if is_mimo_val_out else 0.0
                    logs[f'val_{m.get_name()}'] = m_val

            for cb in all_callbacks: cb.on_epoch_end(epoch, logs)
            
            if verbose:
                if verbose == 1:
                    if validation_data:
                        val_msg = f" - val_loss: {logs['val_loss']:.4f}"
                        for m in self.metrics:
                            name = m.get_name()
                            val_msg += f" - val_{name}: {logs[f'val_{name}']:.4f}"
                        print(val_msg)
                else:
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
        if self.inputs is not None:
            # Functional API forward pass
            tensor_map = {}
            
            # Map input values
            if isinstance(self.inputs, list):
                for i, t in enumerate(self.inputs):
                    tensor_map[id(t)] = inputs[i]
            else:
                tensor_map[id(self.inputs)] = inputs
            
            from ..layers.core.input_layer import InputLayer
            for node in self._nodes_ordered:
                if isinstance(node.layer, InputLayer):
                    continue
                
                # Prepare inputs for this node
                if isinstance(node.input_tensors, list):
                    node_inputs = [tensor_map.get(id(t)) for t in node.input_tensors]
                else:
                    node_inputs = tensor_map.get(id(node.input_tensors))
                
                output = node.layer.forward(node_inputs, training=training)
                
                # Capture state AFTER forward so it captures the current call's data
                node.state = self._capture_layer_state(node.layer)
                
                # Store outputs
                if isinstance(node.output_tensors, list):
                    for i, t in enumerate(node.output_tensors):
                        tensor_map[id(t)] = output[i]
                else:
                    tensor_map[id(node.output_tensors)] = output

            
            # Return model outputs
            if isinstance(self.outputs, list):
                return [tensor_map[id(o)] for o in self.outputs]
            else:
                return tensor_map[id(self.outputs)]

        # Sequential or Subclassed forward pass
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
        if self.outputs is not None:
            # Functional API backward pass
            grad_map = {}
            
            # Map output gradients
            if isinstance(self.outputs, list):
                for i, t in enumerate(self.outputs):
                    grad_map[id(t)] = grad[i]
            else:
                grad_map[id(self.outputs)] = grad
            
            # Initialize accumulators for shared layers
            layer_grads_accumulator = {}
                
            from ..layers.core.input_layer import InputLayer
            for node in reversed(self._nodes_ordered):
                if isinstance(node.layer, InputLayer):
                    continue
                    
                # Get gradients for this node's outputs
                if isinstance(node.output_tensors, list):
                    node_grad_outputs = [grad_map.get(id(t)) for t in node.output_tensors]
                else:
                    node_grad_outputs = grad_map.get(id(node.output_tensors))
                
                if node_grad_outputs is None:
                    continue
                
                # Restore state for this node recursively
                if hasattr(node, 'state'):
                    self._restore_layer_state(node.layer, node.state)
                    
                # Call layer.backward
                # Temporarily clear layer.grads to capture only gradients for this node
                original_grads = node.layer.grads
                node.layer.grads = {}
                
                grad_inputs = node.layer.backward(node_grad_outputs)
                
                # Accumulate parameter gradients
                l_id = id(node.layer)
                if l_id not in layer_grads_accumulator:
                    layer_grads_accumulator[l_id] = {}
                
                for k, v in node.layer.grads.items():
                    if k in layer_grads_accumulator[l_id]:
                        layer_grads_accumulator[l_id][k] += v
                    else:
                        layer_grads_accumulator[l_id][k] = v
                
                # Restore the combined gradients to the layer
                node.layer.grads = layer_grads_accumulator[l_id]
                
                # Propagate gradients to inputs
                if isinstance(node.input_tensors, list):
                    for i, t in enumerate(node.input_tensors):
                        t_id = id(t)
                        if t_id in grad_map:
                            grad_map[t_id] += grad_inputs[i]
                        else:
                            grad_map[t_id] = grad_inputs[i]
                elif node.input_tensors is not None:
                    t_id = id(node.input_tensors)
                    if t_id in grad_map:
                        grad_map[t_id] += grad_inputs
                    else:
                        grad_map[t_id] = grad_inputs
            
            # Return gradients for model inputs
            if isinstance(self.inputs, list):
                return [grad_map.get(id(i)) for i in self.inputs]
            else:
                return grad_map.get(id(self.inputs))

        # Sequential or Subclassed backward pass
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def compute_output_shape(self, input_shape):
        if self.outputs is not None:
            if isinstance(self.outputs, list):
                return [o.shape for o in self.outputs]
            return self.outputs.shape
        
        # For Sequential models
        if not self.layers:
            return input_shape
        
        curr_shape = input_shape
        for layer in self.layers:
            curr_shape = layer.compute_output_shape(curr_shape)
        return curr_shape

    def build(self, input_shape):
        if self.inputs is not None:
            # Functional models are built during construction
            self.input_shape = input_shape
            self.built = True
            return

        # For subclassed models with custom forward, skip sequential build
        # They manage their own layer building internally
        if type(self).forward is not Model.forward:
            self.input_shape = input_shape
            self.built = True
            return

        # For Sequential models, build layers in sequence
        self.input_shape = input_shape
        curr_shape = input_shape
        for layer in self.layers:
            layer.build(curr_shape)
            curr_shape = layer.compute_output_shape(curr_shape)
        self.built = True

    def predict(self, x):
        return self.forward(x, training=False)

    def evaluate(self, x, y):
        is_mimo_out = isinstance(self.outputs, list)
        output = self.predict(x)
        if is_mimo_out:
            loss = sum(self.loss_fn(y[j], output[j]) for j in range(len(self.outputs)))
        else:
            loss = self.loss_fn(y, output)
        results = {'loss': loss}
        for m in self.metrics:
            try:
                m_val = m(y, output)
            except (TypeError, ValueError):
                m_val = m(y[0], output[0]) if is_mimo_out else 0.0
            results[m.get_name()] = m_val
        return results

    def summary(self):
        """
        Prints a Keras-style summary of the model.
        """
        is_functional = self.inputs is not None
        
        print("-" * 85)
        if is_functional:
            print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10} {'Connected to':<25}")
        else:
            print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 85)
        
        total_params = 0
        trainable_params = 0
        
        curr_shape = None
        if self.layers and self.layers[0].input_shape:
            curr_shape = self.layers[0].input_shape

        for layer in self.layers:
            name = layer.name or layer.__class__.__name__
            layer_type = layer.__class__.__name__
            
            # Use layer's own input/output shapes if built
            if layer.built:
                try:
                    output_shape = layer.compute_output_shape(layer.input_shape)
                except Exception:
                    output_shape = "multiple"
            elif curr_shape is not None:
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
            
            if is_functional:
                # Find which layers this layer is connected to via _inbound_nodes
                connected_to = []
                if hasattr(layer, '_inbound_nodes'):
                    for node in layer._inbound_nodes:
                        # Only consider nodes that belong to this model's execution path
                        if node in self._nodes_ordered:
                            if isinstance(node.input_tensors, list):
                                for t in node.input_tensors:
                                    if t.node:
                                        connected_to.append(t.node.layer.name or t.node.layer.__class__.__name__)
                            elif node.input_tensors and node.input_tensors.node:
                                connected_to.append(node.input_tensors.node.layer.name or node.input_tensors.node.layer.__class__.__name__)
                
                connected_str = ", ".join(connected_to) if connected_to else ""
                print(f"{name + ' (' + layer_type + ')':<25} {str(output_shape):<20} {params:<10,} {connected_str:<25}")
            else:
                print(f"{name + ' (' + layer_type + ')':<25} {str(output_shape):<20} {params:<10,}")
            
        print("=" * 85)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("-" * 85)

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
