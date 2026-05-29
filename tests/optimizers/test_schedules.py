import numpy as np
import pytest
from neutro.optimizers.schedules import ExponentialDecay, InverseTimeDecay

def test_exponential_decay():
    initial_lr = 0.1
    decay_steps = 100
    decay_rate = 0.9
    
    lr_schedule = ExponentialDecay(initial_lr, decay_steps, decay_rate)
    
    assert lr_schedule(0) == initial_lr
    assert lr_schedule(100) == initial_lr * decay_rate
    assert lr_schedule(200) == initial_lr * (decay_rate ** 2)

def test_inverse_time_decay():
    initial_lr = 0.1
    decay_steps = 100
    decay_rate = 0.5
    
    lr_schedule = InverseTimeDecay(initial_lr, decay_steps, decay_rate)
    
    assert lr_schedule(0) == initial_lr
    # lr = initial_lr / (1 + decay_rate * (step / decay_steps))
    assert lr_schedule(100) == initial_lr / (1 + decay_rate)
    assert lr_schedule(200) == initial_lr / (1 + 2 * decay_rate)
