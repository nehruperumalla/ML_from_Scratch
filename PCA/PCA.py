import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray) -> None:
        # Mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # Covariance
        cov = np.cov(X.T)
        # Eigenvalues, Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        # Sort Eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # Get first n_components of sorted eigenvectors
        self.components = eigenvectors[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X - self.mean
        return np.dot(X, self.components)
