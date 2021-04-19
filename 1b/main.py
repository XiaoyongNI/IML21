# -*- coding: utf-8 -*-

from sklearn.linear_model import Ridge
import numpy as np

# Data loading
data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
y = data[:, 1]
X = data[:, 2:7]

X = X[0:50, :]
y = y[0:50]


X_expanded = np.zeros((np.size(X,0), 21))
X_expanded[:, 0:4] = X[:,0:4]
X_expanded[:, 20] = X_expanded[:,20] + 1

for i in range(5):
    X_expanded[:,i+5] = X[:,i] ** 2
    X_expanded[:,i+10] = np.exp(X[:,i])
    X_expanded[:,i+15] = np.cos(X[:,i])


lambda_0 = 1
ridge = Ridge(alpha=lambda_0, fit_intercept=False)
ridge.fit(X_expanded, y)
print(ridge.coef_)

np.savetxt('new.csv', ridge.coef_, delimiter = ',')