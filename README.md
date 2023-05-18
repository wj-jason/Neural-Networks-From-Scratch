# Writing a Neural Network from Scratch!

This personal project will cover the journey it takes it create a 'simple' multilayer perceptron using C++ <br>
The final product neural net will be a binary classification network with the following architecture:<br>
> input (2 nodes) -> hidden (10 nodes) -> hidden (10 nodes) -> output (1 node)

The dataset we will be using is sklearn.datasets.make_circles<br>
This dataset will require the use of non-linearity, introduced with ReLU functions between the hidden layers. 
<br>
This project will be split into three phases:
1. Mathematical foundations
2. Translating the math into C++
3. Testing new C++ model against pre-existing PyTorch model with the same architecture
