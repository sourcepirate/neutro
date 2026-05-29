import numpy as np

class LearningRateSchedule:
    def __call__(self, step):
        raise NotImplementedError

class ExponentialDecay(LearningRateSchedule):
    """
    A LearningRateSchedule that uses an exponential decay schedule.
    """
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase=False):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, step):
        p = step / self.decay_steps
        if self.staircase:
            p = np.floor(p)
        return self.initial_learning_rate * (self.decay_rate ** p)

class InverseTimeDecay(LearningRateSchedule):
    """
    A LearningRateSchedule that uses an inverse time decay schedule.
    """
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase=False):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, step):
        p = step / self.decay_steps
        if self.staircase:
            p = np.floor(p)
        return self.initial_learning_rate / (1 + self.decay_rate * p)
