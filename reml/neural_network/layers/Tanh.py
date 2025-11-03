from reml.activation import dtanh, tanh

from .Layer import Layer


class Tanh(Layer):
    def __init__(self):
        self._x = None

    def forward(self, x):
        self._x = x
        return tanh(x)

    def backward(self, grad_output):
        return grad_output * dtanh(self._x)
