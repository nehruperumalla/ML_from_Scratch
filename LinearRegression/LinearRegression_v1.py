import numpy as np


class LinearRegression:
    def __init__(self, lr: float = 0.01, n_iters: float = 1000):
        self.n_iters = n_iters
        self.lr = lr
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Inserting X0 Column with 1's for scalar multiplication with Bias
        X = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X.shape

        # Initializing Slope and intercept a.k.a weights & biases
        self.theta = np.ones(n_features)

        # Vectorized Gradient Descent
        for _ in range(self.n_iters):
            self.theta -= self.lr * \
                ((1 / n_samples) * X.T @ (X @ self.theta - y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Inserting X0 Column with 1's for scalar multiplication with Bias
        X = np.insert(X, 0, 1, axis=1)

        # Returns predictions on test data
        return X @ self.theta
