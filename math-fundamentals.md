# Demystifying Neural Networks

A neural network is simply a function. <br>
In our case, the neural network can be defined as a function $F$, that takes an input $\vec{X}$, where $\vec{X}$ is a $2\text{x}1$ column vector. Then, $F(\vec{X}) \in [0, 1]$, where $0$ and $1$ are each assiged to one of the two classes (remember, we're building a binary classification network). <br>
<br>
First, we will break down the outline for this section, before diving into what everything means:
1. Forward pass
2. Activation functions
3. Loss function
4. Backpropagation
5. Gradient descent

---

## Forward Pass

The forward pass of a network refers to the process in which the inputs are _passed_ into the network to produce some output; wether that output be correct or not. Simply put, the act of inputting our $\vec{X}$ into $F$ is the forward pass. <br>
<br>
So what happens? <br>
<br>
Let's break it down into a simpler case, with just one node connected to another.<br>
```
IPNUT_NODE --> OUTPUT_NODE
```
