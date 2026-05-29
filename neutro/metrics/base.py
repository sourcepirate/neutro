class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError
    def get_name(self):
        raise NotImplementedError
