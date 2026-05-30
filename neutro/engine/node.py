import numpy as np

class KerasTensor:
    """
    Symbolic representation of a tensor in the functional API.
    """
    def __init__(self, shape, node=None, name=None):
        self.shape = shape
        self.node = node  # The node that produced this tensor
        self.name = name

    def __repr__(self):
        return f"KerasTensor(shape={self.shape}, name={self.name})"

class Node:
    """
    Represents a 'call' to a layer.
    Connects input KerasTensors to output KerasTensors.
    """
    def __init__(self, layer, input_tensors, output_tensors):
        self.layer = layer
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        
        # Register the node in the layer
        if not hasattr(layer, '_inbound_nodes'):
            layer._inbound_nodes = []
        layer._inbound_nodes.append(self)
        
        # Link output tensors to this node
        if isinstance(output_tensors, list):
            for t in output_tensors:
                t.node = self
        else:
            output_tensors.node = self

    def __repr__(self):
        return f"Node(layer={self.layer.name or self.layer.__class__.__name__})"
