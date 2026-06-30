import numpy as np
from neutro.autograd import Tensor, GradientTape, Function


class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_data['x']
        return 2 * x * grad_output


def test_square():
    with GradientTape() as tape:
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        tape.watch(x)
        y = Square.apply(x)
        loss = y.sum()
    tape.gradient(loss, [x])
    assert np.allclose(x.grad, 2 * x.data)


class Scale(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale=scale)
        return x * scale

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.saved_data['scale']
        return grad_output * scale


def test_scale():
    with GradientTape() as tape:
        x = Tensor(np.array([1.0, 2.0]))
        tape.watch(x)
        y = Scale.apply(x, 3.0)
        loss = y.sum()
    tape.gradient(loss, [x])
    assert np.allclose(x.grad, 3.0)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_data['a']
        b = ctx.saved_data['b']
        return grad_output * b, grad_output * a


def test_multi_input():
    with GradientTape() as tape:
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0]))
        tape.watch(a)
        tape.watch(b)
        y = Mul.apply(a, b)
        loss = y.sum()
    tape.gradient(loss, [a, b])
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)


class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.exp(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_data['x']
        return np.exp(x) * grad_output


def test_exp():
    with GradientTape() as tape:
        x = Tensor(np.array([0.0, 1.0, 2.0]))
        tape.watch(x)
        y = Exp.apply(x)
        loss = y.sum()
    tape.gradient(loss, [x])
    assert np.allclose(x.grad, np.exp(x.data))


def test_no_watch():
    x = Tensor(np.array([1.0, 2.0]))
    y = Square.apply(x)
    assert y.grad is None
