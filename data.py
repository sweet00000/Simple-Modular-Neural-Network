import numpy as np
from numpy import genfromtxt

ds = np.genfromtxt("data\iris.csv", float, delimiter=',', skip_header=1)

data = ds[:,0:4]
true = ds[:,4]

X = np.reshape(data, (150,4,1))
Y = np.reshape(true, (150,1,1))