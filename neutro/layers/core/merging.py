import numpy as np
from ..base import Layer

class Add(Layer):
    """
    Layer that adds a list of inputs.
    It takes as input a list of tensors, all of the same shape, 
    and returns a single tensor (also of the same shape).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape here would be a list of shapes if we support multiple inputs
        # But in our current Model.forward, we pass single inputs.
        # For Add/Concatenate to work in a Sequential model, it's tricky.
        # Usually they are used in functional API or as internal blocks.
        if isinstance(input_shape, list):
            self.input_shape = input_shape
            self.output_shape = input_shape[0]
        else:
            self.input_shape = input_shape
            self.output_shape = input_shape
        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def forward(self, inputs, training=False):
        """
        inputs: list of ndarrays
        """
        self.input_lengths = len(inputs)
        return sum(inputs)

    def backward(self, grad_output):
        # The gradient of the sum is the gradient itself for each input
        return [grad_output for _ in range(self.input_lengths)]

class Concatenate(Layer):
    """
    Layer that concatenates a list of inputs.
    It takes as input a list of tensors, all of the same shape except 
    for the concatenation axis, and returns a single tensor that is 
    the concatenation of all inputs.
    """
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            return input_shape
        
        # Calculate output shape based on concatenation axis
        out_shape = list(input_shape[0])
        concat_dim = 0
        for shape in input_shape:
            # Handle None in shapes (symbolic)
            dim = shape[self.axis]
            if dim is None:
                concat_dim = None
                break
            concat_dim += dim
        
        out_shape[self.axis] = concat_dim
        return tuple(out_shape)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        self.input_shapes = [i.shape for i in inputs]
        return np.concatenate(inputs, axis=self.axis)

    def backward(self, grad_output):
        # Split grad_output along the same axis
        indices = np.cumsum([s[self.axis] for s in self.input_shapes])[:-1]
        return np.split(grad_output, indices, axis=self.axis)

class Multiply(Layer):
    """
    Layer that multiplies (element-wise) a list of inputs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        self.inputs = inputs
        res = inputs[0].copy()
        for i in range(1, len(inputs)):
            res *= inputs[i]
        return res

    def backward(self, grad_output):
        grads = []
        for i in range(len(self.inputs)):
            # Grad for input i is product of all other inputs * grad_output
            g = grad_output.copy()
            for j in range(len(self.inputs)):
                if i == j: continue
                g *= self.inputs[j]
            grads.append(g)
        return grads

class Average(Layer):
    """
    Layer that computes the average (element-wise) of a list of inputs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        self.input_lengths = len(inputs)
        return sum(inputs) / self.input_lengths

    def backward(self, grad_output):
        return [grad_output / self.input_lengths for _ in range(self.input_lengths)]

class Maximum(Layer):
    """
    Layer that computes the maximum (element-wise) of a list of inputs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        self.inputs = inputs
        res = inputs[0].copy()
        for i in range(1, len(inputs)):
            res = np.maximum(res, inputs[i])
        return res

    def backward(self, grad_output):
        # Gradient goes to the input that was the maximum
        max_val = self.forward(self.inputs)
        grads = []
        for inp in self.inputs:
            mask = (inp == max_val)
            grads.append(grad_output * mask)
        return grads

class Minimum(Layer):
    """
    Layer that computes the minimum (element-wise) of a list of inputs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        self.inputs = inputs
        res = inputs[0].copy()
        for i in range(1, len(inputs)):
            res = np.minimum(res, inputs[i])
        return res

    def backward(self, grad_output):
        min_val = self.forward(self.inputs)
        grads = []
        for inp in self.inputs:
            mask = (inp == min_val)
            grads.append(grad_output * mask)
        return grads
