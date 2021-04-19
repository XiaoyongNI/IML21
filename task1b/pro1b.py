#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

# Data loading
data = np.loadtxt("train_1b.csv", delimiter=",", skiprows=1)
y = data[:, 1]
X = data[:, 2:7]

# ϕ(X)
X_new = np.zeros((700, 21))
X_new[:, 0:5] = X[:, 0:5]

for i in range(5):#i=0,1,2,3,4
    X_new[:,i+5] = X[:,i] ** 2
    X_new[:,i+10] = np.exp(X[:,i])
    X_new[:,i+15] = np.cos(X[:,i])

X_new[:, 20] = X_new[:, 20] + 1

#training
reg = RidgeCV(alphas=(0.1,1.0,10), fit_intercept=False, cv=10)
reg.fit(X_new, y)
print(reg.score(X_new, y))

np.savetxt('new.csv', reg.coef_, delimiter = ',')

