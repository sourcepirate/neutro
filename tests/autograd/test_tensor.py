import numpy as np
from neutro.autograd import Tensor, GradientTape


def test_tensor_creation():
    t = Tensor(np.array([1.0, 2.0, 3.0]))
    assert t.grad is None
    assert t.shape == (3,)


def test_add():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        tape.watch(a)
        tape.watch(b)
        c = a + b
    tape.gradient(c, [a, b])
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, 1.0)


def test_add_scalar():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0]))
        tape.watch(a)
        c = a + 5.0
    tape.gradient(c, [a])
    assert np.allclose(a.grad, 1.0)


def test_radd():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0]))
        tape.watch(a)
        c = 5.0 + a
    tape.gradient(c, [a])
    assert np.allclose(a.grad, 1.0)


def test_sub():
    with GradientTape() as tape:
        a = Tensor(np.array([5.0, 6.0]))
        b = Tensor(np.array([1.0, 2.0]))
        tape.watch(a)
        tape.watch(b)
        c = a - b
    tape.gradient(c, [a, b])
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, -1.0)


def test_mul():
    with GradientTape() as tape:
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0]))
        tape.watch(a)
        tape.watch(b)
        c = a * b
    tape.gradient(c, [a, b])
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)


def test_neg():
    with GradientTape() as tape:
        a = Tensor(np.array([2.0, -3.0]))
        tape.watch(a)
        c = -a
    tape.gradient(c, [a])
    assert np.allclose(a.grad, -1.0)


def test_pow():
    with GradientTape() as tape:
        a = Tensor(np.array([2.0, 3.0]))
        tape.watch(a)
        c = a ** 2
    tape.gradient(c, [a])
    assert np.allclose(a.grad, 2 * a.data)


def test_matmul_2d():
    with GradientTape() as tape:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        tape.watch(a)
        tape.watch(b)
        c = a @ b
    tape.gradient(c, [a, b])
    expected_ga = np.ones_like(a.data) @ b.data.T
    expected_gb = a.data.T @ np.ones_like(b.data)
    assert np.allclose(a.grad, expected_ga)
    assert np.allclose(b.grad, expected_gb)


def test_sum_grad():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        tape.watch(a)
        c = a.sum()
    tape.gradient(c, [a])
    assert np.allclose(a.grad, 1.0)


def test_mean_grad():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        tape.watch(a)
        c = a.mean()
    tape.gradient(c, [a])
    assert np.allclose(a.grad, 0.25)


def test_chain_rule():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0]))
        tape.watch(a)
        b = a + a
        c = b * 3.0
    tape.gradient(c, [a])
    assert np.allclose(a.grad, 6.0)


def test_accumulate_on_shared():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0]))
        tape.watch(a)
        b = a * 2.0
        c = a * 3.0
        d = b + c
    tape.gradient(d, [a])
    assert np.allclose(a.grad, 5.0)


def test_zero_grad():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0]))
        tape.watch(a)
        c = a * 2.0
    tape.gradient(c, [a])
    assert a.grad is not None
    a.zero_grad()
    assert a.grad is None


def test_reshape():
    with GradientTape() as tape:
        a = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        tape.watch(a)
        b = a.reshape(3, 2)
        c = b.sum()
    tape.gradient(c, [a])
    assert a.grad.shape == a.shape
    assert np.allclose(a.grad, 1.0)


def test_transpose():
    with GradientTape() as tape:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        tape.watch(a)
        b = a.T
        c = b.sum()
    tape.gradient(c, [a])
    assert a.grad.shape == a.shape
    assert np.allclose(a.grad, 1.0)


def test_slice():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        tape.watch(a)
        b = a[1:3]
        c = b.sum()
    tape.gradient(c, [a])
    assert np.allclose(a.grad, [0.0, 1.0, 1.0, 0.0])


def test_broadcast_add():
    with GradientTape() as tape:
        a = Tensor(np.array([[1.0, 2.0, 3.0]]))
        b = Tensor(np.array([[4.0], [5.0]]))
        tape.watch(a)
        tape.watch(b)
        c = a + b
        loss = c.sum()
    tape.gradient(loss, [a, b])
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    assert np.allclose(a.grad, [[2.0, 2.0, 2.0]])
    assert np.allclose(b.grad, [[3.0], [3.0]])


def test_no_watch_no_grad():
    with GradientTape() as tape:
        a = Tensor(np.array([1.0, 2.0]))
        b = a * 2.0  # a not watched
    grads = tape.gradient(b, [a])
    assert a.grad is None


def test_sqrt():
    with GradientTape() as tape:
        a = Tensor(np.array([4.0, 9.0]))
        tape.watch(a)
        b = a.sqrt()
    tape.gradient(b, [a])
    assert np.allclose(a.grad, 1.0 / (2.0 * np.sqrt(np.array([4.0, 9.0]))))
