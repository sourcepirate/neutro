import copy
import inspect

import joblib
import numpy as np

from .. import losses as losses_module
from .. import metrics as metrics_module
from .. import optimizers as optimizers_module
from ..engine.node import Node
from ..layers.attention.kv_cache import KVCache
from ..layers.base import Layer
from ..layers.core.input_layer import InputLayer
from neutro.autograd import Tensor as AutoTensor, as_tensor


def _iter_layer_tree(root):
    """Yield each layer in `root`'s layer tree exactly once, cycle-safe."""
    stack = [root]
    seen = set()
    while stack:
        layer = stack.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        yield layer
        stack.extend(reversed(layer.sublayers))


class Model(Layer):
    """Base class for trainable models.

    Three construction styles are supported:

    - ``Sequential([...])``: a linear stack of layers.
    - ``Model(inputs=..., outputs=...)``: the Functional API (a DAG).
    - Subclassing ``Model`` and overriding ``forward``/``backward``.

    A model owns an optimizer, a loss function, and metrics after ``compile``.
    """

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
        """Traverse the graph from outputs to inputs to discover layers and nodes."""
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
        """Attach an optimizer, a loss, and metrics to the model.

        Args:
            optimizer: An ``Optimizer`` instance or a name: ``"sgd"``,
                ``"adam"``, or ``"adamw"``.
            loss: A loss instance or a name understood by
                ``neutro.losses.get`` (``"mse"``,
                ``"categorical_crossentropy"``,
                ``"sparse_categorical_crossentropy"``).
            metrics: Optional list of metric names or instances.
        """
        # Accept both Keras-style names ("adam") and optimizer instances.
        self.optimizer = optimizers_module.get(optimizer)
        if isinstance(loss, str):
            self.loss_fn = losses_module.get(loss)
        else:
            self.loss_fn = loss
        self.metrics = [metrics_module.get(m) for m in (metrics or [])]

    def _get_all_layers(self, layers=None, visited=None):
        roots = self.layers if layers is None else layers
        seen = set() if visited is None else visited
        stack = list(reversed(roots))
        result = []
        while stack:
            layer = stack.pop()
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            result.append(layer)
            stack.extend(reversed(layer.sublayers))
        return result

    _STATE_EXCLUDE = {'params', 'grads', 'built', 'input_shape', 'output_shape',
                      'name', '_inbound_nodes', 'trainable'}

    @staticmethod
    def _capture_layer_state(layer):
        """Recursively capture state of a layer and all its sublayers.

        State is deep-copied so that later mutations during subsequent
        forward passes (e.g. shared layers across branches) cannot corrupt
        the captured snapshot.

        Returns dict: {id(sublayer): {attr_name: value, ...}}
        """
        return {
            id(l): {k: copy.deepcopy(v) for k, v in l.__dict__.items()
                    if k not in Model._STATE_EXCLUDE}
            for l in _iter_layer_tree(layer)
        }

    @staticmethod
    def _restore_layer_state(layer, state):
        """Restore state captured by _capture_layer_state onto layer tree."""
        for l in _iter_layer_tree(layer):
            for k, v in state.get(id(l), {}).items():
                setattr(l, k, v)

    @staticmethod
    def _clear_layer_grads(layer):
        for l in _iter_layer_tree(layer):
            l.grads = {}

    @staticmethod
    def _accumulate_layer_grads(layer, grads_accumulator):
        for l in _iter_layer_tree(layer):
            layer_acc = grads_accumulator.setdefault(id(l), {})
            for k, v in l.grads.items():
                if k in layer_acc:
                    layer_acc[k] += v
                else:
                    layer_acc[k] = np.array(v, copy=True)
            l.grads = layer_acc

    @staticmethod
    def _is_mimo(outputs):
        return isinstance(outputs, list)

    def _compute_loss(self, y, output):
        if self._is_mimo(self.outputs):
            return sum(self.loss_fn(y[j], output[j]) for j in range(len(self.outputs)))
        return self.loss_fn(y, output)

    @staticmethod
    def _eval_metric(metric, y, output, is_mimo_out):
        try:
            return metric(y, output)
        except (TypeError, ValueError):
            return metric(y[0], output[0]) if is_mimo_out else 0.0

    def _metric_summary(self, logs, prefix=''):
        return ''.join(
            f" - {prefix}{m.get_name()}: {logs[f'{prefix}{m.get_name()}']:.4f}"
            for m in self.metrics)

    def fit(self, x, y=None, epochs=1, batch_size=32, verbose=1, validation_data=None, callbacks=None):
        """Train the model.

        ``x`` may be a NumPy array, a list of arrays (MIMO), or a generator
        yielding ``(x_batch, y_batch)``. Each layer's ``backward`` is called;
        the base implementation of ``backward`` is tape-backed.

        Returns:
            A ``History`` object with ``history['loss']`` and one list per
            metric.
        """
        return self._fit_loop(
            x=x, y=y, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data=validation_data, callbacks=callbacks, use_autograd=False)

    def autograd_fit(self, x, y=None, epochs=1, batch_size=32, verbose=1, validation_data=None, callbacks=None):
        """Train with one ``GradientTape`` around the whole forward + loss.

        This is an alternative to :meth:`fit`. It is useful for seeing the
        entire computation as a single graph, but ``fit`` is the default and
        recommended path.
        """
        return self._fit_loop(
            x=x, y=y, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data=validation_data, callbacks=callbacks, use_autograd=True)

    def _fit_loop(self, x, y=None, epochs=1, batch_size=32, verbose=1,
                  validation_data=None, callbacks=None, use_autograd=False):
        """Delegate the epoch/batch loop to :mod:`neutro.models.training`."""
        from .training import fit_loop
        return fit_loop(
            self, x=x, y=y, epochs=epochs, batch_size=batch_size,
            verbose=verbose, validation_data=validation_data,
            callbacks=callbacks, use_autograd=use_autograd)

    def _train_on_batch(self, x_batch, y_batch, use_autograd):
        """Delegate one training step to :mod:`neutro.models.training`."""
        from .training import train_on_batch
        return train_on_batch(self, x_batch, y_batch, use_autograd)

    def _update_validation_logs(self, logs, validation_data):
        """Delegate validation evaluation to :mod:`neutro.models.training`."""
        from .training import update_validation_logs
        return update_validation_logs(self, logs, validation_data)

    @property
    def trainable_params(self):
        params = []
        for layer in self._get_all_layers():
            if getattr(layer, 'trainable', True):
                for p in layer.params.values():
                    if isinstance(p, AutoTensor):
                        params.append(p)
        return params

    def _zero_all_grads(self):
        for p in self.trainable_params:
            p.zero_grad()

    def forward(self, inputs, training=False, kv_cache=None):
        """Run inputs through the model.

        Functional models execute their recorded nodes in topological order.
        Sequential and subclassed models call ``self.layers`` in order, passing
        ``kv_cache`` to layers that accept it.
        """
        if self.inputs is not None:
            # Functional API forward pass
            tensor_map = {}

            # Map input values
            if isinstance(self.inputs, list):
                for i, t in enumerate(self.inputs):
                    tensor_map[id(t)] = inputs[i]
            else:
                tensor_map[id(self.inputs)] = inputs

            for node in self._nodes_ordered:
                if isinstance(node.layer, InputLayer):
                    continue

                # Prepare inputs for this node
                if isinstance(node.input_tensors, list):
                    node_inputs = [tensor_map.get(id(t)) for t in node.input_tensors]
                else:
                    node_inputs = tensor_map.get(id(node.input_tensors))

                output = node.layer.forward(node_inputs, training=training)
                node.layer._last_inputs = as_tensor(node_inputs)
                node.layer._last_kwargs = {'training': training}

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
            if kv_cache is not None and 'kv_cache' in inspect.signature(layer.forward).parameters:
                # Layer accepts kv_cache (Attention or Blocks)
                inputs = layer(inputs, training=training, kv_cache=kv_cache, layer_id=i)
            else:
                inputs = layer(inputs, training=training)
        return inputs

    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        """Autoregressive generation with KV Caching."""
        cache = KVCache()

        # Start with the full prompt
        curr_tokens = start_tokens
        generated = start_tokens

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(curr_tokens, training=False, kv_cache=cache)

            # Get the logits for the last token only: (batch, seq, vocab) -> (batch, vocab)
            next_token_logits = logits[:, -1, :] / temperature

            # Softmax probabilities
            probs = np.exp(next_token_logits - np.max(next_token_logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)

            # Sample next token
            next_token = np.array([np.random.choice(len(p), p=p) for p in probs])
            next_token = next_token.reshape(-1, 1)

            # With KV Cache, only the last token needs to be passed forward
            curr_tokens = next_token
            generated = np.concatenate([generated, next_token], axis=1)

        return generated

    def backward(self, grad):
        """Propagate ``grad`` backward through the model.

        Functional models walk their nodes in reverse topological order and
        accumulate gradients for shared layers. Sequential models call each
        layer's ``backward`` in reverse order.
        """
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
                # Temporarily clear layer-tree grads to capture only this node call
                self._clear_layer_grads(node.layer)
                grad_inputs = node.layer.backward(node_grad_outputs)

                # Accumulate parameter gradients across the full layer tree
                self._accumulate_layer_grads(node.layer, layer_grads_accumulator)

                # Propagate gradients to inputs
                if isinstance(node.input_tensors, list):
                    if not isinstance(grad_inputs, list):
                        grad_inputs = [grad_inputs]
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
        """Run inference and return the model output (a ``Tensor``)."""
        return self.forward(x, training=False)

    @staticmethod
    def _to_scalar(v):
        if isinstance(v, AutoTensor):
            return float(v.data)
        return float(v)

    def evaluate(self, x, y):
        """Compute loss and metrics on ``(x, y)``.

        Returns:
            ``{"loss": float, "<metric>": float, ...}``.
        """
        is_mimo_out = self._is_mimo(self.outputs)
        output = self.predict(x)
        loss = self._compute_loss(y, output)
        results = {'loss': self._to_scalar(loss)}
        for m in self.metrics:
            m_val = self._eval_metric(m, y, output, is_mimo_out)
            results[m.get_name()] = self._to_scalar(m_val) if not isinstance(m_val, float) else m_val
        return results

    def summary(self):
        """Print a Keras-style layer/parameter table."""
        from .summary import print_summary
        print_summary(self)

    def save(self, filepath):
        """Serialize the whole model with joblib."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """Load a model previously written by :meth:`save`."""
        return joblib.load(filepath)


class Sequential(Model):
    """A linear stack of layers.

    ``Sequential([Dense(16), Dense(3)])`` calls each layer's output as the next
    layer's input. ``build`` is triggered by the first layer's ``input_shape``
    or by the first real call.
    """

    def __init__(self, layers=None):
        super().__init__()
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """Append ``layer`` and build it from the previous layer's shape."""
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
                elif len(shape) == 3:  # (h, w, c)
                    shape = (None,) + shape
                layer.build(shape)
        else:
            # Build based on previous layer's output shape
            prev_layer = self.layers[-1]
            if prev_layer.built:
                input_shape = prev_layer.compute_output_shape(prev_layer.input_shape)
                layer.build(input_shape)

        self.layers.append(layer)
