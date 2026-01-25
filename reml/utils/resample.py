import numpy as np


def bootstrap_sampling(X, y, random_state=42):
    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)
    idxs = rng.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)


def he_init(fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float64) * std