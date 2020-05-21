import numpy as np
from copy import deepcopy
from des_knn import *
from sklearn.model_selection import train_test_split
import cProfile

dataset = "./datasets/liver"
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

des = DES_KNN()

des.fit(X_train, y_train)
# des.predict(X_test)
cProfile.run('des.predict(X_test)')