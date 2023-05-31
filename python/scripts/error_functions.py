import numpy as np

# mean squared error
def MSE(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# derivative of mean squared error
def MSE_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
