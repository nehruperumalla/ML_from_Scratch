import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from LogisticRegression import LogisticRegression
from sklearn.metrics import confusion_matrix

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)

print(X_train.shape, y_train.shape)

clf = LogisticRegression(lr=0.001, n_iters=5000)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)


def accuracy(y_test, y_preds):
    return np.sum(y_preds == y_test) / len(y_test)


print(accuracy(y_test, predictions))
print(confusion_matrix(y_test, predictions))
