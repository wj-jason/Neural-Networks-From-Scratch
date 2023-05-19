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
Let's break it down into a simpler case, with just one node connected to another.<br><br>
<img src=images/math-foundations/two-nodes.png width=500>
<br>
Each node, labeled $X$ for input, and $Y$ for output, simply hold numbers. As the information in $X$ moves into $Y$, it undergoes a transformation, given as $Y=WX+b$.<br>
In our context, $W$ is the _weight_ associated with the connection between $X$ and $Y$, and $b$ is the _bias_ associated with $Y$. <br>
Every connection has a unique weight, and every node (except input nodes) has a unique bias. Referencing the neural net we are working up to, we have nodes layered in the following order:
```
2 -> 10 -> 10 -> 1
```
Thus, we will have 21 biases, and 130 weights. <br>
These are our _learnable parameters_, that is, the purpose of our neural network will be to optimize every single one of these terms to produce an output that we desire. 
<br>
Now this is great and all, but at the scale neural nets operate, we would need hundreds of thousands, even millions of these tiny linear equations to describe just the input outputs of a single layer. Furthermore, we have no way of clearly referencing a particular node, weight, or bias, in any particular layer. Thus we need to determine a method for indexing every learnable parameter, as well as a concise way to represent the inputs and outputs of each layer. <br>
<br>
We will now introduce the notation that will be used for the remainder of the project.
Instead of individual node inputs and outpus being represented with $X$ and $Y$, we will reserve these values for the inputs and outputs of the _entire_ network.<br>
We will represent a node with $a$ (maybe for activation?), along with the follwing super/subscripts: $a_n^{(L)}$. <br>
> _a_: some node<br>
> _n_: the index of the node within the layer<br>
> _L_: the current layer<br>

<img src=images/math-foundations/node-indexing.png width=500>

Moving on to the weights, an indiviudal weight will now be represented with $w_{k, n}^{(L)}$ (note lowercase). <br>
> _w_: some weight<br>
> _k_: index of neuron in _next_ layer<br>
> _n_: index of node within _current_ layer<br>
> _L_: the layer the weight is connecting to ($L\neq 0$)
<img src=images/math-foundations/weight-indexing.png width=500>
<br>
The weights for some layer $L$ are then stored in a matrix $W_L$, where each vertical entry corresponds to the weights associated with an input node.<br>
<img src=images/math-foundations/weight-matrix.png width=500>

