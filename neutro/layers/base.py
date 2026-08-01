import numpy as np


class Layer:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.trainable = True
        self.built = False
        self.params = {}
        self.grads = {}
        self.input_shape = kwargs.get('input_shape')
        self.output_shape = None
        self._inbound_nodes = []

    def build(self, input_shape):
        self.input_shape = input_shape
        self.built = True

    @property
    def sublayers(self):
        """Return all nested layers contained within this layer.

        Traverses instance attributes, including sublayers stored in lists,
        tuples, or dictionaries, with cycle protection and de-duplication.
        """
        layers = []
        visited_ids = {id(self)}
        stack = []
        for attr in vars(self).values():
            stack.append(attr)

        while stack:
            attr = stack.pop()
            if isinstance(attr, Layer):
                if id(attr) not in visited_ids:
                    visited_ids.add(id(attr))
                    layers.append(attr)
                    for nested in vars(attr).values():
                        stack.append(nested)
            elif isinstance(attr, (list, tuple, set)):
                stack.extend(attr)
            elif isinstance(attr, dict):
                stack.extend(attr.values())
        return layers

    def count_params(self):
        """Count the total number of parameters in this layer and its sublayers."""
        from neutro.autograd import Tensor as AutoTensor
        count = 0
        for p in self.params.values():
            count += p.data.size if isinstance(p, AutoTensor) else p.size
        for layer in self.sublayers:
            count += layer.count_params()
        return count

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Should be overridden by subclasses that transform the input shape.
        """
        if hasattr(self, 'output_shape') and self.output_shape is not None:
            return self.output_shape
        return input_shape

    def _collect_tensor_params(self):
        """Gather all autograd Tensor parameters across the layer tree."""
        from neutro.autograd import Tensor as AT
        tensor_params = {}
        stack = [self]
        while stack:
            layer = stack.pop()
            for param_name, param_value in layer.params.items():
                if isinstance(param_value, AT):
                    tensor_params[(id(layer), param_name)] = param_value
            stack.extend(layer.sublayers)
        return tensor_params

    def _captured_inputs(self):
        """Build autograd Tensor input(s) from the last forward call."""
        from neutro.autograd import Tensor as AT
        inputs = getattr(self, '_last_inputs', None)
        if inputs is None:
            return None
        if isinstance(inputs, list):
            return [AT(i) if not isinstance(i, AT) else i for i in inputs]
        return inputs if isinstance(inputs, AT) else AT(np.asarray(inputs))

    def backward(self, grad_output):
        from neutro.autograd import GradientTape, Tensor as AT

        t_inputs = self._captured_inputs()
        if t_inputs is None:
            return np.asarray(grad_output)

        fwd_args = getattr(self, '_last_args', ())
        fwd_kwargs = getattr(self, '_last_kwargs', {}).copy()
        # KV cache and layer id are runtime bookkeeping, not part of the
        # differentiable computation.
        fwd_kwargs.pop('kv_cache', None)
        fwd_kwargs.pop('layer_id', None)

        all_sources = list(self._collect_tensor_params().values())
        if isinstance(t_inputs, list):
            all_sources.extend(t_inputs)
        else:
            all_sources.append(t_inputs)

        with GradientTape() as tape:
            for source in all_sources:
                tape.watch(source)
            output = self.forward(t_inputs, *fwd_args, **fwd_kwargs)
            g_t = AT(np.asarray(grad_output))
            if isinstance(output, list):
                loss = sum((o * g).sum() for o, g in zip(output, g_t))
            else:
                loss = (output * g_t).sum()

        tape.gradient(loss, all_sources)

        from neutro.autograd import Tensor as AT2
        stack_layers = [self]
        while stack_layers:
            layer = stack_layers.pop()
            for param_name, param_value in layer.params.items():
                if isinstance(param_value, AT2) and param_value.grad is not None:
                    layer.grads[param_name] = param_value.grad.copy()
            stack_layers.extend(layer.sublayers)

        if isinstance(t_inputs, list):
            return [t.grad.copy() if t.grad is not None else None for t in t_inputs]
        return t_inputs.grad.copy() if t_inputs.grad is not None else None

    def _is_symbolic_input(self, inputs):
        from ..engine.node import KerasTensor

        if isinstance(inputs, KerasTensor):
            return True
        return (isinstance(inputs, list)
                and any(isinstance(i, KerasTensor) for i in inputs))

    def _symbolic_call(self, inputs):
        """Execute the layer in graph-building (Functional API) mode."""
        from ..engine.node import KerasTensor, Node

        if isinstance(inputs, list):
            input_shapes = [i.shape for i in inputs]
        else:
            input_shapes = inputs.shape

        if not self.built:
            self.build(input_shapes)

        output_shape = self.compute_output_shape(input_shapes)

        if isinstance(output_shape, list):
            output_tensors = [KerasTensor(shape=s) for s in output_shape]
        else:
            output_tensors = KerasTensor(shape=output_shape)

        Node(self, input_tensors=inputs, output_tensors=output_tensors)
        return output_tensors

    def _eager_call(self, inputs, args, kwargs):
        """Execute the layer in eager (Sequential/manual) mode."""
        from neutro.autograd import as_tensor as _as_tensor

        t_inputs = _as_tensor(inputs)
        if not self.built:
            if isinstance(t_inputs, list):
                self.build([i.shape for i in t_inputs])
            else:
                self.build(t_inputs.shape)
        self._last_inputs = t_inputs
        self._last_args = args
        self._last_kwargs = kwargs
        output = self.forward(t_inputs, *args, **kwargs)
        self._last_output = output
        return output

    def __call__(self, inputs, *args, **kwargs):
        if self._is_symbolic_input(inputs):
            return self._symbolic_call(inputs)
        return self._eager_call(inputs, args, kwargs)

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads
