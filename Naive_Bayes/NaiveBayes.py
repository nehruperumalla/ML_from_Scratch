import numpy as np


class NaiveBayes:
    def __init__(self):
        self._classes = None
        self._mean = {}
        self._var = {}
        self._priors = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self._classes = np.unique(y)

        # Calculating the Mean, Varience and priors on the training data
        for idx, c in enumerate(self._classes):
            X_c = X[c == y]
            self._mean[idx] = X_c.mean(axis=0)
            self._var[idx] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / n_samples

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Prediction of Classes on test data
        return np.array([self._predict(x) for x in X])

    def _predict(self, x: np.ndarray) -> int:
        # Calculating the probabailities of all classes using Posteriors and Priors
        posteriors = [np.sum(np.log(self._pdf(idx, x))) + np.log(self._priors[idx])
                      for idx, c in enumerate(self._classes)]
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, idx: int, x: np.ndarray) -> float:
        # Using the Guassian Distribution for Class conditional probability
        mean = self._mean[idx]
        var = self._var[idx]
        numerator = np.exp(-(x - mean) ** 2/(2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
