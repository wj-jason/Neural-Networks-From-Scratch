from linear import Linear
from sigmoid import Sigmoid
from mse import MSE, MSE_prime
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch

samples = 1000

X_, y_ = make_circles(samples, noise=0.03, random_state=42)

X = torch.from_numpy(X_).type(torch.float)
y = torch.from_numpy(y_).type(torch.float)

print(X_[0])
print(y_[0])

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=42)

network = [
    Linear(2, 10),
    Sigmoid(),
    Linear(10, 10),
    Sigmoid(),
    Linear(10, 1)
]

epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    error = 0
    for x, y in zip(X, y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += MSE(y, output)
        grad = MSE_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    error /= len(X)
    print('%d%d, error = %f' % (epoch + 1, epochs, error))

