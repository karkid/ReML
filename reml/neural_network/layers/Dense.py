import numpy as np

from .Layer import Layer


def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)


def he_init(fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float64) * std


class Dense(Layer):
    def __init__(self, in_features, out_features, init="xavier", bias=True):
        if init == "xavier":
            self.W = xavier_init(in_features, out_features)
        elif init == "he":
            self.W = he_init(in_features, out_features)
        else:
            self.W = np.random.randn(in_features, out_features).astype(np.float64)
            self.W *= 0.01
        self.b = np.zeros((1, out_features), dtype=np.float64) if bias else None
        self._x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        pass

    def forward(self, x):
        self._x = x
        return x @ self.W + (self.b if self.b is not None else 0)

    def backward(self, grad_output):
        self.dW = self._x.T @ grad_output
        if self.b is not None:
            self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

    def params_and_grads(self):
        if self.W is not None:
            yield self.W, self.dW
        if self.b is not None:
            yield self.b, self.db
