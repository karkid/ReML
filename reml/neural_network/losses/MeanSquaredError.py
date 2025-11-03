import numpy as np

from .Loss import Loss


class MeanSquaredError(Loss):
    def __init__(self):
        self._y_pred = None
        self._y_true = None

    def forward(self, y_pred, y_true):
        self._y_pred, self._y_true = y_pred, y_true
        if y_pred.size == 0:
            return 0.0
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        n = self._y_pred.shape[0]
        if n == 0:
            return np.array([])
        return 2 * (self._y_pred - self._y_true) / n
