from ..base import Layer
from ...engine.node import KerasTensor, Node

class InputLayer(Layer):
    """
    Entry point into a computation graph (like Keras InputLayer).
    No trainable params — just passes inputs through as the root of the graph.
    """
    def __init__(self, input_shape=None, name=None, **kwargs):
        super().__init__(name=name, input_shape=input_shape, **kwargs)
        if input_shape is not None:
            self.build(input_shape)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.built = True

    def forward(self, inputs, training=False):
        return inputs

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
