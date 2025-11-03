import numpy as np

from reml.utils.decorators import auto_repr, check_fitter


@auto_repr
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.scale_ = None
        self.is_fitted = False

    def fit(self, X, y=None):
        X = np.array(X)
        n_samples = X.shape[0]

        # Suppress warnings for mean and std calculations on edge cases
        with np.errstate(all="ignore"):
            self.mean_ = np.mean(X, axis=0)

            if n_samples <= 1:
                # For single sample or empty data, std will be NaN or 0
                self.std_ = np.std(X, axis=0, ddof=1)
            else:
                self.std_ = np.std(X, axis=0, ddof=1)

        # Handle zeros and NaN in scale for transform method
        self.scale_ = np.where((self.std_ == 0) | np.isnan(self.std_), 1.0, self.std_)
        self.is_fitted = True
        return self

    @check_fitter
    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
