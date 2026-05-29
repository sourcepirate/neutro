class Optimizer:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, value):
        self.learning_rate = value

    def step(self, layers):
        raise NotImplementedError
