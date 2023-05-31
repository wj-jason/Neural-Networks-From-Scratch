import numpy as np

class Sigmoid:

    def __init__(self):

        self.weights = None

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        
        self.sigmoid = sigmoid
        self.sigmoid_prime = sigmoid_prime
    
    def forward(self, input):
        self.input = input
        return self.sigmoid(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.sigmoid_prime(self.input))
