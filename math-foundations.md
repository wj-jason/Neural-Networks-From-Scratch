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
> _L_: the layer the weight is connecting to ($L\neq 0$)<br>

<img src=images/math-foundations/weight-indexing.png width=500>

The weights for some layer $L$ are then stored in a matrix $W_L$, where each vertical entry corresponds to the weights associated with an input node.<br>

<img src=images/math-foundations/weight-matrix.png width=500>

Note that some sources will instead store each input node weight horizontally, not vertically, which requires a transpose operation further down the line. To avoid that, we will store them vertically.
<br>
<br>
Finally, we have the biases. As stated earlier, every node (except input nodes) will have a bias term associated with them. In order to store these, we will do so in a column vector. The column vector will be represented with $B_L$, where once again, $L$ is the reference to the layer at which the biases operate. <br>

<img src=images/math-foundations/bias-indexing.png width=500>

Now that we have all our notation figured out, let's see what happens to the output of a node with multiple connections: 

<img src=images/math-foundations/multiple-nodes.png width=500>

Instead of just one weight and one activation value affecting the output node, we have three weights, and three activations, along with a single bias that will determine the value of the output node. Mathematically, we can represent this process as a weighted sum:<br>
$$\Large a_0^{(1)}=(w_{0,0}^{(1)}a_0^{(0)}+w_{0,1}^{(1)}a_1^{(0)}+w_{0,2}^{(1)}a_2^{(0)})+b_0^{(1)}$$
But what if there are multiple output nodes:

<img src=images/math-foundations/multiple-outputs.png width=500>

Then, we can compute $a_0^{(1)}$ and $a_1^{(1)}$ indivudally, based on their respective connections:
$$\Large a_0^{(1)}=(w_{0,0}^{(1)}a_0^{(0)}+w_{0,1}^{(1)}a_1^{(0)}+w_{0,2}^{(1)}a_2^{(0)})+b_0^{(1)}$$
$$\Large a_1^{(1)}=(w_{1,0}^{(1)}a_0^{(0)}+w_{1,1}^{(1)}a_1^{(0)}+w_{1,2}^{(1)}a_2^{(0)})+b_1^{(1)}$$
As you can probably tell by now, this gets really messy, really quickly. To avoid having to compute each node indiviudally, we can store the outputs of an entire layer in a column vector, and the computation then comes down to the matrix vector product of the predefined weight matrix $W_L$ and a column vector of input nodes. Then we can add the bias column vector to obtain the result. 

<img src=images/math-foundations/multiple-outputs-transform.png width=750>

As you can see, the result of this linear transformation is the same as iteratively computing individual node outputs! We now have a powerful way to represent the inputs and outputs of each layer. <br>
We can generalize this result, relating the $L-1^{th}$ layer to the $L^{th}$ as follows:

<img src=images/math-foundations/generalized-input-outputs.png width=750>

For one last notational reminder, $X_L$ represents the values of the activations within some layer $L$<br>
<br>
All in all, we can now represent the values passing between two layers with $X_L=W_LX_{L-1}+B_L$.<br>
We know our network has 4 layers, an input, output, and two hidden layers. Representing the _forward pass_ then comes down to layering these expressions together.<br>
Representing our neural network as a function $F$ with an input $X$, the forward pass is:
$$\Large F(X)=W_2(W_1(W_0X+B_0)+B_1)+B_2$$
As a final note, consider the dimensionality of the input as it moves through the function. Upon multiplication with $W_0$, a $10\text{x}2$ matrix, the result is a $10\text{x}1$ column vector, that we can then legally add the $10\text{x}1$ bias vector to. This trend holds, and as we move into a layer with $k$ nodes, the resulting output vector lives in $\mathbb{R}^k$ space.
