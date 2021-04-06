import numpy as np
from collections import Counter

class KNN:
	def __init__(self, k = 3):
		self.k = k

	def euclidean_distance(self, x1, x2):
		return np.sqrt(np.sum((x1 - x2) ** 2))

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		predicted_labels = [self._predict(x) for x in X]
		return predicted_labels

	def _predict(self, x):
		distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
		k_indices = np.argsort(distances)[:self.k]
		k_nearest_labels = [self.y_train[index] for index in k_indices]
		return Counter(k_nearest_labels).most_common(1)[0][0]