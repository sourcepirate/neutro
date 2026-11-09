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
        result = []
        for root in roots:
            for layer in _iter_layer_tree(root):
                if id(layer) in seen:
                    continue
                seen.add(id(layer))
                result.append(layer)
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

    def _seed_tensor_map(self, inputs):
        if isinstance(self.inputs, list):
            return {id(t): value for t, value in zip(self.inputs, inputs)}
        return {id(self.inputs): inputs}

    @staticmethod
    def _node_inputs(node, tensor_map):
        if isinstance(node.input_tensors, list):
            return [tensor_map.get(id(t)) for t in node.input_tensors]
        return tensor_map.get(id(node.input_tensors))

    @staticmethod
    def _store_node_outputs(node, output, tensor_map):
        if isinstance(node.output_tensors, list):
            for tensor, value in zip(node.output_tensors, output):
                tensor_map[id(tensor)] = value
        else:
            tensor_map[id(node.output_tensors)] = output

    def _forward_functional(self, inputs, training):
        tensor_map = self._seed_tensor_map(inputs)

        for node in self._nodes_ordered:
            if isinstance(node.layer, InputLayer):
                continue

            node_inputs = self._node_inputs(node, tensor_map)
            output = node.layer.forward(node_inputs, training=training)
            node.layer._last_inputs = as_tensor(node_inputs)
            node.layer._last_kwargs = {'training': training}
            node.state = self._capture_layer_state(node.layer)
            self._store_node_outputs(node, output, tensor_map)

        if isinstance(self.outputs, list):
            return [tensor_map[id(tensor)] for tensor in self.outputs]
        return tensor_map[id(self.outputs)]

    def _forward_layers(self, inputs, training, kv_cache):
        for i, layer in enumerate(self.layers):
            accepts_cache = 'kv_cache' in inspect.signature(layer.forward).parameters
            if kv_cache is not None and accepts_cache:
                inputs = layer(inputs, training=training, kv_cache=kv_cache, layer_id=i)
            else:
                inputs = layer(inputs, training=training)
        return inputs

    def forward(self, inputs, training=False, kv_cache=None):
        """Run inputs through the model.

        Functional models execute their recorded nodes in topological order.
        Sequential and subclassed models call ``self.layers`` in order, passing
        ``kv_cache`` to layers that accept it.
        """
        if self.inputs is not None:
            return self._forward_functional(inputs, training)
        return self._forward_layers(inputs, training, kv_cache)

    @staticmethod
    def _sample_next_token(logits, temperature):
        scores = np.asarray(logits)[:, -1, :] / temperature
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        samples = [np.random.choice(len(row), p=row) for row in probs]
        return np.asarray(samples).reshape(-1, 1)

    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        """Autoregressive generation with KV Caching."""
        cache = KVCache()
        current = start_tokens
        generated = start_tokens

        for _ in range(max_new_tokens):
            logits = self.forward(current, training=False, kv_cache=cache)
            next_token = self._sample_next_token(logits, temperature)
            current = next_token
            generated = np.concatenate([generated, next_token], axis=1)

        return generated

    def _seed_grad_map(self, grad):
        if isinstance(self.outputs, list):
            return {id(tensor): value for tensor, value in zip(self.outputs, grad)}
        return {id(self.outputs): grad}

    @staticmethod
    def _node_output_grads(node, grad_map):
        if isinstance(node.output_tensors, list):
            return [grad_map.get(id(t)) for t in node.output_tensors]
        return grad_map.get(id(node.output_tensors))

    @staticmethod
    def _add_grad(grad_map, tensor, grad):
        tensor_id = id(tensor)
        if tensor_id in grad_map:
            grad_map[tensor_id] = grad_map[tensor_id] + grad
        else:
            grad_map[tensor_id] = grad

    def _propagate_input_grads(self, node, grad_inputs, grad_map):
        if isinstance(node.input_tensors, list):
            if not isinstance(grad_inputs, list):
                grad_inputs = [grad_inputs]
            for tensor, grad in zip(node.input_tensors, grad_inputs):
                self._add_grad(grad_map, tensor, grad)
        elif node.input_tensors is not None:
            self._add_grad(grad_map, node.input_tensors, grad_inputs)

    def _backward_functional(self, grad):
        grad_map = self._seed_grad_map(grad)
        layer_grads = {}

        for node in reversed(self._nodes_ordered):
            if isinstance(node.layer, InputLayer):
                continue

            output_grads = self._node_output_grads(node, grad_map)
            if output_grads is None:
                continue

            if hasattr(node, 'state'):
                self._restore_layer_state(node.layer, node.state)

            self._clear_layer_grads(node.layer)
            input_grads = node.layer.backward(output_grads)
            self._accumulate_layer_grads(node.layer, layer_grads)
            self._propagate_input_grads(node, input_grads, grad_map)

        if isinstance(self.inputs, list):
            return [grad_map.get(id(tensor)) for tensor in self.inputs]
        return grad_map.get(id(self.inputs))

    def _backward_layers(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def backward(self, grad):
        """Propagate ``grad`` backward through the model.

        Functional models walk their nodes in reverse topological order and
        accumulate gradients for shared layers. Sequential models call each
        layer's ``backward`` in reverse order.
        """
        if self.outputs is not None:
            return self._backward_functional(grad)
        return self._backward_layers(grad)

    def compute_output_shape(self, input_shape):
        if self.outputs is not None:
            return [t.shape for t in self.outputs] if isinstance(self.outputs, list) else self.outputs.shape

        shape = input_shape
        for layer in self.layers:
            shape = layer.compute_output_shape(shape)
        return shape

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

    @staticmethod
    def _with_batch_dim(shape):
        return (None,) + tuple(shape) if len(shape) in (1, 3) else tuple(shape)

    def add(self, layer):
        """Append ``layer`` and build it from the previous layer's shape."""
        if not self.layers:
            if layer.input_shape:
                layer.build(self._with_batch_dim(layer.input_shape))
        else:
            previous = self.layers[-1]
            if previous.built:
                layer.build(previous.compute_output_shape(previous.input_shape))

        self.layers.append(layer)
