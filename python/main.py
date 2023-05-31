from layers import Linear, Sigmoid
from error_functions import MSE, MSE_prime
from run import run
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np

# import & format data
samples = 1000
X_raw, Y_raw = make_circles(samples, noise=0.03, random_state=42)
X = np.reshape(X_raw, (samples, 2, 1))
Y = np.reshape(Y_raw, (samples, 1, 1))

# split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# intialize the model
network = [
    Linear(2, 10),
    Sigmoid(),
    Linear(10, 10),
    Sigmoid(),
    Linear(10, 1),
    Sigmoid()
]

# hyperparams
epochs = 250
learning_rate = 0.1

# run!
run(network, X_train, Y_train, MSE, MSE_prime, epochs, learning_rate)
