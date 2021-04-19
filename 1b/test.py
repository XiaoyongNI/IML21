import sklearn.linear_model as skl
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

X = np.zeros((2, 21))

X[:, 0:4] = np.ones((2,4))
# X[:, 0:4] = 1
print(X)