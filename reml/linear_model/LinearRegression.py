import numpy as np

from reml.utils.decorators import auto_repr, check_fitter


@auto_repr
class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iteration=1000):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.weights = None
        self.bias = None
        self.is_fitted = False
        self.losses = []  # tracking of loss over iterations

    def _as_2d(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _as_1d(self, y):
        y = np.asarray(y, dtype=float).ravel()
        return y

    def fit(self, X, y):
        X = self._as_2d(X)
        y = self._as_1d(y)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        # Using half-MSE loss so gradients are (1/n) * X^T (y_hat - y)
        for _ in range(self.n_iteration):
            y_pred = X @ self.weights + self.bias

            # gradients
            error = y_pred - y
            dw = (1.0 / n_samples) * (X.T @ error)
            db = (1.0 / n_samples) * np.sum(error)

            # Clip gradients to prevent overflow
            max_grad_norm = 1e6
            dw = np.clip(dw, -max_grad_norm, max_grad_norm)
            db = np.clip(db, -max_grad_norm, max_grad_norm)

            # update with overflow protection
            weight_update = self.learning_rate * dw
            bias_update = self.learning_rate * db

            # Check for potential overflow
            if np.any(np.isnan(weight_update)) or np.any(np.isinf(weight_update)):
                break
            if np.isnan(bias_update) or np.isinf(bias_update):
                break

            self.weights -= weight_update
            self.bias -= bias_update

            # track half-MSE (matches gradient convention) with overflow protection
            error_squared = error**2
            if np.any(np.isinf(error_squared)) or np.any(np.isnan(error_squared)):
                loss = np.inf
            else:
                loss = 0.5 * np.mean(error_squared)
            self.losses.append(loss)

        self.is_fitted = True
        return self

    @check_fitter
    def predict(self, X):
        X = self._as_2d(X)
        return X @ self.weights + self.bias
