import numpy as np
import pytest
from neutro.layers.core.merging import Add, Concatenate, Multiply, Average, Maximum, Minimum


class TestMultiply:
    def test_forward(self):
        layer = Multiply()
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        out = layer.forward([a, b])
        expected = a * b
        np.testing.assert_array_equal(out, expected)

    def test_backward(self):
        layer = Multiply()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        layer.forward([a, b])
        grad = np.array([[1.0, 1.0], [1.0, 1.0]])
        grads = layer.backward(grad)
        assert len(grads) == 2
        np.testing.assert_array_equal(grads[0], b)
        np.testing.assert_array_equal(grads[1], a)

    def test_compute_output_shape(self):
        layer = Multiply()
        shape = layer.compute_output_shape([(None, 32), (None, 32)])
        assert shape == (None, 32)

    def test_compute_output_shape_single(self):
        layer = Multiply()
        shape = layer.compute_output_shape((16, 32))
        assert shape == (16, 32)


class TestAverage:
    def test_forward_and_backward(self):
        layer = Average()
        a = np.array([[1.0, 3.0], [5.0, 7.0]])
        b = np.array([[2.0, 4.0], [6.0, 8.0]])
        out = layer.forward([a, b])
        expected = (a + b) / 2
        np.testing.assert_array_equal(out, expected)

        grad = np.array([[1.0, 1.0], [1.0, 1.0]])
        grads = layer.backward(grad)
        assert len(grads) == 2
        np.testing.assert_array_equal(grads[0], grad / 2)
        np.testing.assert_array_equal(grads[1], grad / 2)

    def test_compute_output_shape(self):
        layer = Average()
        assert layer.compute_output_shape([(None, 32), (None, 32)]) == (None, 32)
        assert layer.compute_output_shape((16, 32)) == (16, 32)


class TestMaximum:
    def test_forward_maximum(self):
        layer = Maximum()
        a = np.array([[1.0, 5.0], [3.0, 2.0]])
        b = np.array([[4.0, 2.0], [1.0, 6.0]])
        out = layer.forward([a, b])
        expected = np.maximum(a, b)
        np.testing.assert_array_equal(out, expected)

    def test_backward_maximum(self):
        layer = Maximum()
        a = np.array([[1.0, 5.0], [3.0, 2.0]])
        b = np.array([[4.0, 2.0], [1.0, 6.0]])
        layer.forward([a, b])
        grad = np.array([[1.0, 1.0], [1.0, 1.0]])
        grads = layer.backward(grad)
        assert len(grads) == 2
        expected_grad_a = np.array([[0.0, 1.0], [1.0, 0.0]])
        expected_grad_b = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(grads[0], expected_grad_a)
        np.testing.assert_array_equal(grads[1], expected_grad_b)

    def test_compute_output_shape(self):
        layer = Maximum()
        assert layer.compute_output_shape([(None, 32), (None, 32)]) == (None, 32)
        assert layer.compute_output_shape((16, 32)) == (16, 32)


class TestMinimum:
    def test_forward_minimum(self):
        layer = Minimum()
        a = np.array([[1.0, 5.0], [3.0, 2.0]])
        b = np.array([[4.0, 2.0], [1.0, 6.0]])
        out = layer.forward([a, b])
        expected = np.minimum(a, b)
        np.testing.assert_array_equal(out, expected)

    def test_backward_minimum(self):
        layer = Minimum()
        a = np.array([[1.0, 5.0], [3.0, 2.0]])
        b = np.array([[4.0, 2.0], [1.0, 6.0]])
        layer.forward([a, b])
        grad = np.array([[1.0, 1.0], [1.0, 1.0]])
        grads = layer.backward(grad)
        assert len(grads) == 2
        expected_grad_a = np.array([[1.0, 0.0], [0.0, 1.0]])
        expected_grad_b = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(grads[0], expected_grad_a)
        np.testing.assert_array_equal(grads[1], expected_grad_b)

    def test_compute_output_shape(self):
        layer = Minimum()
        assert layer.compute_output_shape([(None, 32), (None, 32)]) == (None, 32)
        assert layer.compute_output_shape((16, 32)) == (16, 32)


class TestAddComputeOutputShape:
    def test_compute_output_shape_non_list(self):
        layer = Add()
        assert layer.compute_output_shape((16, 32)) == (16, 32)


class TestConcatenateComputeOutputShape:
    def test_compute_output_shape_non_list(self):
        layer = Concatenate()
        assert layer.compute_output_shape((16, 32)) == (16, 32)
