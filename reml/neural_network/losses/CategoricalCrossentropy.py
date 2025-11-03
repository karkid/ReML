import numpy as np

from .Loss import Loss


class CategoricalCrossentropy(Loss):
    def __init__(self):
        self._y_pred = None
        self._y_true = None

    def forward(self, logits, y_true):
        if logits.size == 0:
            return 0.0
        z = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / np.sum(e, axis=1, keepdims=True)
        self._y_pred = probs
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            idx = y_true.astype(int).reshape(-1)
            loss = -np.mean(np.log(probs[np.arange(len(idx)), idx] + 1e-12))
            onehot = np.zeros_like(probs)
            onehot[np.arange(len(idx)), idx] = 1.0
            self._y_true = onehot
        else:
            self._y_true = y_true
            loss = -np.mean(np.sum(y_true * np.log(probs + 1e-12), axis=1))
        return loss

    def backward(self):
        n = self._y_pred.shape[0]
        if n == 0:
            return np.array([])
        return (self._y_pred - self._y_true) / n
