from objects import *
from functions import *
from data import *

network = [
    Dense(4, 10),
    Hypertan(),

    Dense(10, 10),
    Hypertan(),
    
    Dense(10, 1),
    Hypertan()
]

train(network, mse, dmse, X, Y, 1000, .001)