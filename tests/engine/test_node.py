import numpy as np
from neutro.engine.node import KerasTensor, Node


class FakeLayer:
    def __init__(self):
        self.name = "fake_layer"


def test_keras_tensor_repr():
    t = KerasTensor(shape=(None, 32, 32, 3), name="input")
    r = repr(t)
    assert "KerasTensor" in r
    assert "(None, 32, 32, 3)" in r
    assert "input" in r


def test_node_single_output():
    layer = FakeLayer()
    output = KerasTensor(shape=(None, 10), name="output")
    node = Node(layer, input_tensors=[], output_tensors=output)

    assert node.layer is layer
    assert node.output_tensors is output
    assert output.node is node
    assert layer._inbound_nodes == [node]


def test_node_list_output():
    layer = FakeLayer()
    out1 = KerasTensor(shape=(None, 5))
    out2 = KerasTensor(shape=(None, 3))
    node = Node(layer, input_tensors=[], output_tensors=[out1, out2])

    assert node.output_tensors == [out1, out2]
    assert out1.node is node
    assert out2.node is node


def test_node_repr():
    layer = FakeLayer()
    output = KerasTensor(shape=(None, 10))
    node = Node(layer, input_tensors=[], output_tensors=output)
    r = repr(node)
    assert "Node" in r
    assert "fake_layer" in r
