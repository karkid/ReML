import numpy as np

from .Loss import Loss


class BinaryCrossentropy(Loss):
    def __init__(self, eps=1e-12):
        self._y_pred = None
        self._y_true = None
        self.eps = eps

    def forward(self, y_pred, y_true):
        if y_pred.size == 0:
            return 0.0
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        self._y_pred, self._y_true = y_pred, y_true
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self):
        n = self._y_pred.shape[0]
        if n == 0:
            return np.array([])
        return (
            -(self._y_true / self._y_pred - (1 - self._y_true) / (1 - self._y_pred)) / n
        )
