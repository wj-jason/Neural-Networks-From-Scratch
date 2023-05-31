# Math $\rightarrow$ Python

- [`layers.py`](#layerspy)
- [`error_functions.py`](#error_functionspy)
- [`run.py`](#runpy)
- [`main.py`](#mainpy)

## `layers.py`

Starting with `layers.py`, we initalize a base class `Layer` that each subsequent layer can inherit from. This class is abstract, defining forward and backward functions to later be overwritten with the relavent foward pass and backpropagation formulas depending on the type of layer. 

```
class Layer:
    def __init__(self):
        raise NotImplementedError
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_graident, learning_rate):
        raise NotImplementedError
```

We can then create our linear layer. This is the layer that the weights and biases act on, and is called the linear layer as reference to the linear transformation taking place in the forward pass. This layer inherits from `Layer`, overwritting `forward()` and `backward()` with the formulas defined in mathematics.md

```
class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    
    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, np.transpose(self.input))
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * output_gradient
        return np.dot(np.transpose(self.weights), output_gradient)
```

The weights and biases are randomly initalized with it's dimensionality being depedent on the dimensionality of the input and output as expected. The forward pass saves the input as `self.input` so it can be used in `backward()` to compute the input gradient. The backpropagation pass computes the weight gradient by multiplying the output gradient by $X^T$ as shown in mathematics.md, before updating the weights and biases accordingly, and returning the input gradient to be used on the next backpropagation pass.

Moving on, we can define an activation layer that inherits from `Layer` and redefines the forward and backward pass accordingly. The `__init__()` method will take some activation fuction and it's derivative as inputs, meaning we will also need to define the activation function next. 

```
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_graident, learning_rate):
        return np.multiply(output_graident, self.activation_prime(self.input))
```

The activation layer is much simpler than the linear layer, as it's forward pass simply applies a function element-wise on the input vector, and it's backpropagation pass multiplies the output gradient by the derivative of the activation, also element-wise. 

---

## `error_functions.py`

Next, we will need to create our error function, as well as it's derivative. Since we are using MSE (mean squared error), this process is fairly simple:

```
def MSE(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def MSE_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
```

---

## `run.py`

Now we can write the code for the train and test loops. Starting with the train loop, we will first create a function `train()` with the folliwng parameters:
- the model
- inputs
- labeled outputs
- error function
- derivative of error function
- epochs
- learning rate
First, we need to set up a loop to iterate over the range of epochs:

```
for epoch in range(epochs + 1):
```

We now run iterate over the network, computing each output before feeding it back into the next layer:

```
for x, y in zip(X, Y):
    output = x
        for layer in network:
            output = layer.forward(output)
```

On each full pass through, we can compute and sum the error (this isn't necessary but let's us view the model performance), and compute the output gradient using the derivative of the error function:

```
error += error_func(y, output)
grad = error_prime(y, output)
```

With the gradient, we can start the backpropagation pass. We first reverse the network and compute the gradient of each layer using the `backward()` method.

```
for layer in reversed(network):
    grad = layer.backward(grad, learning_rate)
```

This step is technically optional, but without it, we cannot view the accuracy of our model as it trains. First, we take the average error, and print. This code will print the error on each 10th epoch to reduce clutter but this is personal preference.

```
error /= len(X)
if epoch % 10 == 0:
    print(f'Epoch: {epoch:3} | Error: {error}')
```

Put all together, the `train()` method looks like this:

```
def train(network, X, Y, error_func, error_prime, epochs, learning_rate):
    for epoch in range(epochs + 1):
        error = 0
        for x, y in zip(X, Y):
            output = x
            for layer in network:
                output = layer.forward(output)
            error += error_func(y, output)
            grad = error_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(X)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3} | Error: {error}')
```        

The test loop is significantly easier, as there is parameter updating involved. All we need to do is run through each data entry in the test set and compute the forward pass, and error. Since all the code in `test()` can be found in `train()`, there's no need to go over it all again.

```
def test(network, X, Y, error_func):
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += error_func(y, output)
    error /= len(X)
    print(f'Average Error: {error}')
```

From there, we can define another function `run()` combining the train and test:

```
def run(network, X, Y, error_func, error_prime, epochs, learning_rate):
    print()
    
    # training loop
    for epoch in range(epochs + 1):
        error = 0
        for x, y in zip(X, Y):
            output = x
            for layer in network:
                output = layer.forward(output)
            error += error_func(y, output)
            grad = error_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(X)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3} | Error: {error}')
    print('\n---------------------------------------------\n')

    # testing loop
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += error_func(y, output)
    error /= len(X)
    print(f'Test Error: {error}\n')
```

---

## `main.py`

Finally, we can put everything together inside of `main.py`. We start by importing all the objects and functions we defined earlier, along with the dataset and NumPy.

```
from layers import Linear, Sigmoid
from error_functions import MSE, MSE_prime
from run import run
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np
```

Then, we can import and reshape the dataset before splitting it into training and testing groups.

```
samples = 1000
X_raw, Y_raw = make_circles(samples, noise=0.03, random_state=42)
X = np.reshape(X_raw, (samples, 2, 1))
Y = np.reshape(Y_raw, (samples, 1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

We now intialize the model and set our hyperparameters:

```
network = [
    Linear(2, 10),
    Sigmoid(),
    Linear(10, 10),
    Sigmoid(),
    Linear(10, 1),
    Sigmoid()
]

epochs = 250
learning_rate = 0.1
```

Finally calling `run()` on our network as such:

```
run(network, X_train, Y_train, MSE, MSE_prime, epochs, learning_rate)
```

Produces the following output:

![image](https://github.com/wj-jason/Neural-Networks-From-Scratch/assets/116098777/afd24887-620f-4979-b1c2-b8d1ab7b48ec)
