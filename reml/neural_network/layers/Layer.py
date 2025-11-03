class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def params_and_grads(self):
        return []

    def train(self):
        pass

    def eval(self):
        pass
