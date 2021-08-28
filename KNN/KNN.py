import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k: int = 3):
        self.k = k

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        # Euclidean distance calculation between two points with D-Dimensions
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Storing the Train Data
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Class label prediction of test data.
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels

    def _predict(self, x: np.ndarray) -> int:
        # Distance calculation between a query/test point and training data
        distances = [self.euclidean_distance(
            x, x_train) for x_train in self.X_train]

        # Getting the K-Nearest Neighbour's indices of query/test point
        k_indices = np.argsort(distances)[:self.k]

        # Fetching the K-Nearest Neighbour's class labels
        # Returning majority class label among them
        k_nearest_labels = [self.y_train[index] for index in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]
