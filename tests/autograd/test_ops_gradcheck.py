import numpy as np
import pytest
from neutro.autograd import Tensor, GradientTape
from neutro.autograd import ops
from neutro.autograd.utils import numeric_grad, gradclose


def gradcheck(op, x_val, *args, axis=None, rtol=1e-4, atol=1e-6, **kwargs):
    x = Tensor(x_val)
    with GradientTape() as tape:
        tape.watch(x)
        out = op(x, *args, **kwargs) if axis is None else op(x, *args, axis=axis, **kwargs)
        loss = out.sum()
    tape.gradient(loss, [x])

    def func(x_ary):
        t = Tensor(x_ary)
        o = op(t, *args, **kwargs) if axis is None else op(t, *args, axis=axis, **kwargs)
        return o.data.sum()

    numeric = numeric_grad(func, x_val.copy(), eps=1e-5)
    analytic = x.grad
    assert gradclose(analytic, numeric, rtol=rtol, atol=atol), \
        f"gradcheck failed for {op.__name__}: max diff={np.max(np.abs(analytic - numeric))}"


def gradcheck_binary(op, a_val, b_val, rtol=1e-4, atol=1e-6):
    a = Tensor(a_val)
    b = Tensor(b_val)
    with GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        out = op(a, b)
        loss = out.sum()
    tape.gradient(loss, [a, b])

    def func_a(x_ary):
        ta = Tensor(x_ary)
        tb = Tensor(b_val)
        return op(ta, tb).data.sum()

    def func_b(x_ary):
        ta = Tensor(a_val)
        tb = Tensor(x_ary)
        return op(ta, tb).data.sum()

    n_a = numeric_grad(func_a, a_val.copy(), eps=1e-5)
    n_b = numeric_grad(func_b, b_val.copy(), eps=1e-5)

    assert gradclose(a.grad, n_a, rtol=rtol, atol=atol), \
        f"gradcheck failed for {op.__name__} (a): max diff={np.max(np.abs(a.grad - n_a))}"
    assert gradclose(b.grad, n_b, rtol=rtol, atol=atol), \
        f"gradcheck failed for {op.__name__} (b): max diff={np.max(np.abs(b.grad - n_b))}"


def test_grad_relu():
    gradcheck(ops.relu, np.array([-2.0, -1.0, -0.5, 1.0, 2.0]))


def test_grad_sigmoid():
    gradcheck(ops.sigmoid, np.array([-2.0, 0.0, 2.0]))


def test_grad_tanh():
    gradcheck(ops.tanh, np.array([-2.0, 0.0, 2.0]))


def test_grad_silu():
    gradcheck(ops.silu, np.array([-2.0, 0.0, 2.0]))


def test_grad_softmax():
    gradcheck(ops.softmax, np.array([[1.0, 2.0, 3.0]]), axis=-1)


def test_grad_add():
    gradcheck_binary(ops.add, np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))


def test_grad_sub():
    gradcheck_binary(ops.sub, np.array([5.0, 3.0]), np.array([1.0, 2.0]))


def test_grad_mul():
    gradcheck_binary(ops.mul, np.array([2.0, 3.0]), np.array([4.0, 5.0]))


def test_grad_div():
    gradcheck_binary(ops.div, np.array([4.0, 9.0]), np.array([2.0, 3.0]))


def test_grad_pow():
    gradcheck(lambda x: ops._pow(x, 2), np.array([1.0, 2.0, 3.0]))


def test_grad_neg():
    def neg_op(x):
        return ops.neg(x)
    gradcheck(neg_op, np.array([1.0, -2.0, 3.0]))


def test_grad_matmul():
    a_val = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_val = np.array([[5.0, 6.0], [7.0, 8.0]])
    a = Tensor(a_val)
    b = Tensor(b_val)
    with GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        out = ops.matmul(a, b)
        loss = out.sum()
    tape.gradient(loss, [a, b])

    def func_a(x):
        t = Tensor(x)
        tb = Tensor(b_val)
        return ops.matmul(t, tb).data.sum()

    def func_b(x):
        t = Tensor(a_val)
        tb = Tensor(x)
        return ops.matmul(t, tb).data.sum()

    n_a = numeric_grad(func_a, a_val.copy(), eps=1e-5)
    n_b = numeric_grad(func_b, b_val.copy(), eps=1e-5)

    assert gradclose(a.grad, n_a), f"matmul grad a: max diff={np.max(np.abs(a.grad - n_a))}"
    assert gradclose(b.grad, n_b), f"matmul grad b: max diff={np.max(np.abs(b.grad - n_b))}"


def test_grad_sum():
    gradcheck(ops._sum, np.array([[1.0, 2.0], [3.0, 4.0]]), axis=None)
    gradcheck(ops._sum, np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    gradcheck(ops._sum, np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1)


def test_grad_mean():
    gradcheck(ops._mean, np.array([[1.0, 2.0], [3.0, 4.0]]), axis=None)
    gradcheck(ops._mean, np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    gradcheck(ops._mean, np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1)


def test_grad_reshape():
    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def reshape_op(x):
        return ops.reshape(x, (3, 2))

    gradcheck(reshape_op, x_val)


def test_grad_transpose():
    x_val = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def transpose_op(x):
        return ops.transpose(x)

    gradcheck(transpose_op, x_val)


def test_grad_broadcast_add():
    a_val = np.array([[1.0, 2.0, 3.0]])
    b_val = np.array([[4.0], [5.0]])
    a = Tensor(a_val)
    b = Tensor(b_val)
    with GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        out = ops.add(a, b)
        loss = out.sum()
    tape.gradient(loss, [a, b])

    def func_a(x):
        t = Tensor(x)
        return ops.add(t, Tensor(b_val)).data.sum()

    def func_b(x):
        t = Tensor(x)
        return ops.add(Tensor(a_val), t).data.sum()

    n_a = numeric_grad(func_a, a_val.copy(), eps=1e-5)
    n_b = numeric_grad(func_b, b_val.copy(), eps=1e-5)

    assert gradclose(a.grad, n_a, rtol=1e-3), f"broadcast add a: max diff={np.max(np.abs(a.grad - n_a))}"
    assert gradclose(b.grad, n_b, rtol=1e-3), f"broadcast add b: max diff={np.max(np.abs(b.grad - n_b))}"


def test_grad_softmax_stable():
    x_val = np.array([[50.0, 60.0, 100.0]])
    gradcheck(ops.softmax, x_val, axis=-1, rtol=1e-3, atol=1e-4)


def test_grad_sqrt():
    gradcheck(ops.sqrt, np.array([1.0, 4.0, 9.0]))


def test_grad_exp():
    gradcheck(ops.exp, np.array([0.0, 1.0, 2.0]))
