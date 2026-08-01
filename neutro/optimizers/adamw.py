from .adam import Adam, _get_data

DEFAULT_WEIGHT_DECAY = 0.01


class AdamW(Adam):
    """Adam with decoupled weight decay.

    The weight decay is applied directly to the parameter *before* the
    moment updates, decoupled from the gradient (Loshchilov & Hutter, 2019).
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-7, weight_decay=DEFAULT_WEIGHT_DECAY):
        super().__init__(learning_rate=learning_rate, beta_1=beta_1,
                         beta_2=beta_2, epsilon=epsilon)
        self.weight_decay = weight_decay

    def _apply_step(self, layer, param_name, param_value, grad):
        param_data = _get_data(param_value)
        param_data -= self.learning_rate * self.weight_decay * param_data
        super()._apply_step(layer, param_name, param_value, grad)
