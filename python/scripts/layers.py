import numpy as np

# base class
class Layer:
    def __init__(self):
        raise NotImplementedError
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_graident, learning_rate):
        raise NotImplementedError

# applies linear transformation to inputs
# has weights and biases
class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    # forward pass: Y = WX+B
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    
    # backpropagation: compute gradients, update weights and biases, return input gradient
    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, np.transpose(self.input))
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * output_gradient
        return np.dot(np.transpose(self.weights), output_gradient)

# activation layer
# no learnable parameters
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # forward pass: Y = f(X)
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    # backpropagation: compute input gradient
    def backward(self, output_graident, learning_rate):
        return np.multiply(output_graident, self.activation_prime(self.input))

# specific activation fuction
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        def sigmoid_prime(x):
            return (1/(1 + np.exp(-x))) * (1 - (1/(1 + np.exp(-x))))
        
        super().__init__(sigmoid, sigmoid_prime)
