# -*- coding: utf-8 -*-

import sklearn.linear_model as skl
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

# Data loading
data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
y = data[:, 1]
X = data[:, 2:7]

# Computing features ϕ(X)
X_expanded = np.zeros((np.size(X,0), 21))
X_expanded[:, 0:5] = X[:,0:5]
X_expanded[:, 20] = X_expanded[:,20] + 1

for i in range(5):
    X_expanded[:,i+5] = X[:,i] ** 2
    X_expanded[:,i+10] = np.exp(X[:,i])
    X_expanded[:,i+15] = np.cos(X[:,i])

#print(X_expanded.shape)

reg = skl.RidgeCV(alphas=(0.01,1,100), fit_intercept=False, cv=7)# build model
reg.fit(X_expanded, y)# training
print(reg.score(X_expanded, y))

np.savetxt('coe.csv', reg.coef_, delimiter = ',')
