import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2021)

# print(X_train.shape)

from LinearRegression import LinearRegression

regressor = LinearRegression(lr = 0.1)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

def mse(y, y_pred):
	return np.mean((y - y_pred) ** 2)

mse_value = mse(y_test, predictions)

print('MSE: %f' % (mse_value))

y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10, 8))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s = 10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s = 10)
plt.plot(X, y_pred_line, color = 'black', linewidth = 2, label = 'Prediction')
plt.show()
