from reml.activation import dsigmoid, sigmoid

from .Layer import Layer


class Sigmoid(Layer):
    def __init__(self):
        self._x = None

    def forward(self, x):
        self._x = x
        return sigmoid(x)

    def backward(self, grad_output):
        return grad_output * dsigmoid(self._x)
