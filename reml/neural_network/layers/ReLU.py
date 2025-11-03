from reml.activation import drelu, relu

from .Layer import Layer


class ReLU(Layer):
    def __init__(self):
        self._x = None

    def forward(self, x):
        self._x = x
        return relu(x)

    def backward(self, grad_output):
        return grad_output * drelu(self._x)
