from LinearRegression_v1 import LinearRegression
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# class LinearRegression:
#     def __init__(self, lr: float = 0.01, n_iters: float = 1000):
#         self.n_iters = n_iters
#         self.lr = lr
#         self.theta = None
#         self.lis = []

#     def fit(self, X: np.ndarray, y: np.ndarray) -> None:
#         # Inserting X0 Column at 0th Pos with 1's to handle y-intercept
#         X = np.insert(X, 0, 1, axis=1)

#         n_samples, n_features = X.shape

#         # Initializing Slope and intercept a.k.a weights & biases
#         self.theta = np.ones(n_features)

#         # Vectorized Gradient Descent
#         for _ in range(self.n_iters):
#             self.theta -= self.lr * \
#                 ((1 / n_samples) * X.T @ (X @ self.theta - y))
#             self.lis.append(self.theta.copy())

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         # Inserting X0 Column at 0th Pos with 1's to handle y-intercept
#         X = np.insert(X, 0, 1, axis=1)

#         # Returns predictions on test data
#         return X @ self.theta


X, y = datasets.make_regression(
    n_samples=1000, n_features=10, random_state=2021)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)

print(X_train.shape, y_train.shape)
regressor = LinearRegression(lr=0.001, n_iters=10000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


# print(regressor.lis)
# mse_values = [mse(y_train, np.insert(X_train, 0, 1, axis=1) @ theta)
#               for theta in regressor.lis]

mseval = mse(y_test, predictions)

print('MSE:', mseval)
# plt.plot(range(10000), mse_values)
# plt.show()
