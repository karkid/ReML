from reml.activation import dsoftmax, softmax

from .Layer import Layer


class Softmax(Layer):
    def __init__(self):
        self._x = None

    def forward(self, x):
        self._x = x
        return softmax(x)

    def backward(self, grad_output):
        return grad_output * dsoftmax(self._x)
