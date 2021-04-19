# -*- coding: utf-8 -*-

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

# Data loading
data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
y = data[:, 0]
X = data[:, 1:14]

# Split training set and test set
k_fold = 10
kf = KFold(n_splits=k_fold, random_state=None, shuffle=False)

lambda_set = [0.1, 1, 10, 100, 200]
for j in range(len(lambda_set)):
    lambda_0 = lambda_set[j]
    RMSE = 0
    for train_index, test_index in kf.split(X):
        # train
        X_train = X[train_index]
        y_train = y[train_index]

        ridge = Ridge(alpha = lambda_0)
        ridge.fit(X_train, y_train)

        # predict
        X_test = X[test_index]
        y_test = y[test_index]
        y_predicted = ridge.predict(X_test)
        
        # population risk
        MSE = 0
        for i in range(len(y_test)):
            MSE = MSE + pow(y_predicted[i] - y_test[i], 2)
        RMSE = RMSE + pow(MSE / len(y_test), 0.5) 

    RMSE = RMSE / k_fold
    print(RMSE)
