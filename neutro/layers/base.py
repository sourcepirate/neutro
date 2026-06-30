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
        """
        Returns all nested layers within this layer.
        """
        layers = []
        for attr_name in dir(self):
            if attr_name.startswith('_') or attr_name == 'sublayers':
                continue
            try:
                attr = getattr(self, attr_name)
            except AttributeError:
                continue
                
            if isinstance(attr, Layer):
                layers.append(attr)
            elif isinstance(attr, list):
                # Handle nested lists (like in MoELayer experts)
                stack = [attr]
                while stack:
                    curr = stack.pop()
                    for item in curr:
                        if isinstance(item, Layer):
                            layers.append(item)
                        elif isinstance(item, list):
                            stack.append(item)
        return layers

    def count_params(self):
        """
        Counts the total number of parameters in this layer and its sublayers.
        """
        from neutro.autograd import Tensor as AutoTensor
        count = 0
        for p in self.params.values():
            if isinstance(p, AutoTensor):
                count += p.data.size
            else:
                count += p.size
        for layer in self.sublayers:
            count += layer.count_params()
        return count

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        Should be overridden by subclasses.
        """
        if hasattr(self, 'output_shape') and self.output_shape is not None:
            return self.output_shape
        return input_shape

    def backward(self, grad_output):
        from neutro.autograd import Tensor as AT, GradientTape
        inputs = getattr(self, '_last_inputs', None)
        if inputs is None:
            return np.asarray(grad_output)

        if isinstance(inputs, list):
            t_inputs = [AT(i) if not isinstance(i, AT) else i for i in inputs]
        elif not isinstance(inputs, AT):
            t_inputs = AT(np.asarray(inputs))
        else:
            t_inputs = inputs

        fwd_args = getattr(self, '_last_args', ())
        fwd_kwargs = getattr(self, '_last_kwargs', {}).copy()
        fwd_kwargs.pop('kv_cache', None)
        fwd_kwargs.pop('layer_id', None)
        tensor_params = {}
        stack_layers = [self]
        while stack_layers:
            l = stack_layers.pop()
            for pn, pv in l.params.items():
                if isinstance(pv, AT):
                    tensor_params[(id(l), pn)] = pv
            for sl in l.sublayers:
                stack_layers.append(sl)
        all_sources = list(tensor_params.values())
        if isinstance(t_inputs, list):
            all_sources.extend(t_inputs)
        else:
            all_sources.append(t_inputs)

        with GradientTape() as tape:
            for s in all_sources:
                tape.watch(s)
            output = self.forward(t_inputs, *fwd_args, **fwd_kwargs)
            g_t = AT(np.asarray(grad_output))
            if isinstance(output, list):
                loss = sum((o * g).sum() for o, g in zip(output, g_t))
            else:
                loss = (output * g_t).sum()

        tape.gradient(loss, all_sources)

        stack_l = [self]
        while stack_l:
            l = stack_l.pop()
            for pn, pv in l.params.items():
                from neutro.autograd import Tensor as AT2
                if isinstance(pv, AT2) and pv.grad is not None:
                    l.grads[pn] = pv.grad.copy()
            for sl in l.sublayers:
                stack_l.append(sl)

        if isinstance(t_inputs, list):
            return [t.grad.copy() if t.grad is not None else None for t in t_inputs]
        return t_inputs.grad.copy() if t_inputs.grad is not None else None

    def __call__(self, inputs, *args, **kwargs):
        from ..engine.node import KerasTensor, Node

        # Check if inputs are symbolic
        is_symbolic = False
        if isinstance(inputs, KerasTensor):
            is_symbolic = True
        elif isinstance(inputs, list) and any(isinstance(i, KerasTensor) for i in inputs):
            is_symbolic = True
        
        if is_symbolic:
            # Symbolic call (Functional API)
            if isinstance(inputs, list):
                input_shapes = [i.shape for i in inputs]
            else:
                input_shapes = inputs.shape
            
            if not self.built:
                self.build(input_shapes)
            
            output_shape = self.compute_output_shape(input_shapes)
            
            # Create output tensor(s)
            if isinstance(output_shape, list):
                output_tensors = [KerasTensor(shape=s) for s in output_shape]
            else:
                output_tensors = KerasTensor(shape=output_shape)
            
            # Create node
            Node(self, input_tensors=inputs, output_tensors=output_tensors)
            return output_tensors

        # Eager call (Sequential or manual)
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

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads
