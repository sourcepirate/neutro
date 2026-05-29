class Loss:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError
    def gradient(self, y_true, y_pred):
        raise NotImplementedError
