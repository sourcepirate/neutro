import numpy as np
from neutro.initializers import Zeros, Ones, RandomNormal, GlorotUniform, HeNormal, get

def test_zeros():
    z = Zeros()
    assert np.all(z((2, 2)) == 0)

def test_ones():
    o = Ones()
    assert np.all(o((2, 2)) == 1)

def test_random_normal():
    rn = RandomNormal()
    assert rn((10, 10)).shape == (10, 10)

def test_glorot_uniform():
    gu = GlorotUniform()
    assert gu((10, 10)).shape == (10, 10)
    assert gu((3, 3, 1, 8)).shape == (3, 3, 1, 8)
    assert gu((5,)).shape == (5,)

def test_he_normal():
    he = HeNormal()
    assert he((10, 10)).shape == (10, 10)
    assert he((3, 3, 1, 8)).shape == (3, 3, 1, 8)
    assert he((5,)).shape == (5,)

def test_get_initializer():
    assert isinstance(get('zeros'), Zeros)
    assert isinstance(get('he_normal'), HeNormal)
