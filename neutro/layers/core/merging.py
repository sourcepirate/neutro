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

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            self.output_shape = input_shape
        else:
            # Calculate output shape based on concatenation axis
            out_shape = list(input_shape[0])
            concat_dim = 0
            for shape in input_shape:
                concat_dim += shape[self.axis]
            out_shape[self.axis] = concat_dim
            self.output_shape = tuple(out_shape)
        super().build(input_shape)

    def forward(self, inputs, training=False):
        self.input_shapes = [i.shape for i in inputs]
        return np.concatenate(inputs, axis=self.axis)

    def backward(self, grad_output):
        # Split grad_output along the same axis
        indices = np.cumsum([s[self.axis] for s in self.input_shapes])[:-1]
        return np.split(grad_output, indices, axis=self.axis)
