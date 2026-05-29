class Activation:
    def __call__(self, x):
        raise NotImplementedError
    def gradient(self, x):
        raise NotImplementedError
