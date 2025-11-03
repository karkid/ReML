import numpy as np


class SGD:
    def __init__(self, lr=0.1, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, params_and_grads):
        for p, g in params_and_grads:
            if g is None:
                continue
            v = self.velocities.get(id(p), np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            p += v
            self.velocities[id(p)] = v
