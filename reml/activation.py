import numpy as np


def sigmoid(x):
    # Clip x to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    y = sigmoid(x)
    return y * (1.0 - y)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    y = tanh(x)
    return 1.0 - y**2


def relu(x):
    return np.maximum(0, x)


def drelu(x):
    grad = np.zeros_like(x, dtype=float)
    grad[x > 0.0] = 1.0
    return grad


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def dleaky_relu(x, alpha=0.01):
    grad = np.ones_like(x, dtype=float)
    grad[x < 0.0] = alpha
    return grad


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def dsoftplus(x):
    return sigmoid(x)


def softmax(x):
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)  # for numerical stability
    e = np.exp(z)
    return e / np.sum(e)


def dsoftmax(x):
    s = softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))


def delu(x, alpha=1.0):
    grad = np.ones_like(x, dtype=float)
    grad[x < 0.0] = alpha * np.exp(x[x < 0.0])
    return grad
