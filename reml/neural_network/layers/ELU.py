from reml.activation import delu, elu

from .Layer import Layer


class ELU(Layer):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._x = None

    def forward(self, x):
        self._x = x
        return elu(x, self.alpha)

    def backward(self, grad_output):
        return grad_output * delu(self._x, self.alpha)
