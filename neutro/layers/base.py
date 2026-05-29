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
        count = sum(p.size for p in self.params.values())
        for layer in self.sublayers:
            count += layer.count_params()
        return count

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        Should be overridden by subclasses.
        """
        return input_shape
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def __call__(self, inputs, *args, **kwargs):
        if not self.built:
            self.build(inputs.shape)
        return self.forward(inputs, *args, **kwargs)

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads
