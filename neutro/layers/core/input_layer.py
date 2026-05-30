from ..base import Layer
from ...engine.node import KerasTensor, Node

class InputLayer(Layer):
    """
    Layer to be used as an entry point into a Network (a graph of layers).
    """
    def __init__(self, input_shape=None, name=None, **kwargs):
        super().__init__(name=name, input_shape=input_shape, **kwargs)
        if input_shape is not None:
            self.build(input_shape)

    def build(self, input_shape):
        self.input_shape = input_shape
        # Add batch dimension if missing
        if len(input_shape) > 0 and input_shape[0] is not None:
            # We assume users might pass (28, 28, 1) or (None, 28, 28, 1)
            # Keras usually expects input_shape to NOT include batch.
            pass 
        self.built = True

    def forward(self, inputs, training=False):
        return inputs

    def backward(self, grad_output):
        return grad_output

def Input(shape=None, name=None, **kwargs):
    """
    Used to instantiate a Keras tensor.
    """
    if shape is None:
        raise ValueError("Please provide a shape for the Input.")
    
    # Ensure shape is a tuple and starts with None for batch
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    
    # Keras style: if first element is not None, prepend None
    if len(shape) == 0 or shape[0] is not None:
        shape = (None,) + shape

    layer = InputLayer(input_shape=shape, name=name, **kwargs)
    
    # Create the symbolic output tensor
    output_tensor = KerasTensor(shape=shape, name=name)
    
    # Create the node connecting layer to its output
    Node(layer, input_tensors=[], output_tensors=output_tensor)
    
    return output_tensor
