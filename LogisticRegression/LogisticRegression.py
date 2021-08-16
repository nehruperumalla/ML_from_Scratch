import numpy as np


class LogisticRegression:
    def __init__(self, lr: float = 0.1, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Inserting X0 Column with 1's for scalar multiplication with Bias
        X = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X.shape

        # Initializing Slope and intercept a.k.a weights & biases
        self.theta = np.zeros(n_features)

        # Vectorized Gradient Descent
        for _ in range(self.n_iters):
            self.theta -= self.lr * (1 / n_samples) * \
                X.T @ (self.sigmoid(X @ self.theta) - y)

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        # Returns an array of length same as Z with sigmoid function applied to each element of Z,
        # where the resultant of a sigmoid function always lies in range (0,1).
        return 1 / (1 + np.exp(-Z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Inserting X0 Column with 1's for scalar multiplication with Bias
        X = np.insert(X, 0, 1, axis=1)

        # Returns predictions on test data
        # Classifies as True(1) if value is greater than threshold(0.5) else False(0)
        return np.where(self.sigmoid(X @ self.theta) > 0.5, 1, 0)
