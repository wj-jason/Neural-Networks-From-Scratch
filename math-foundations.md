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
$$\Large F(X)=W_3(W_2(W_1X+B_1)+B_2)+B_3$$
As a final note, consider the dimensionality of the input as it moves through the function. Upon multiplication with $W_1$, a $10\text{x}2$ matrix, the result is a $10\text{x}1$ column vector, that we can then legally add the $10\text{x}1$ bias vector to. This trend holds, and as we move into a layer with $k$ nodes, the resulting output vector lives in $\mathbb{R}^k$ space. As obvious as this might seem, it's important to constantly be aware of the dimensionalty of your matricies to ensure there are no errors in your neural network.

---

## Activation Functions

Now for a quick confession, the forward pass formula shown above is not _exactly_ correct. Currenty, our values are dependent soley on the linear transformation that takes place at each step. Furthermore, it was stated earlier that $F(X) \in [0,1]$ but currently, that is not the case.<br>
We will now define two new functions, called ReLU (standing for Rectified Linear Unit), and Sigmoid ($\sigma$).
$$\Large \text{ReLU}(x) = \text{R}(x) = \text{max}(0, x)$$
$$\Large \sigma (x) = \frac{1}{1+e^{-x}}$$
To describe the purpose of each function, ReLU simply eliminates the possibility for negative values, whereas Sigmoid squishes the entire real number line between 0 and 1. <br>
We will introduce the ReLU function at the output of each layer. This introduces non-linearity to our neural network, and solves the 'vanishing gradient' problem, which we will see once we get to backpropagation and gradient descent.<br>
Sigmoid on the other hand is applied to the very end of the network, squishing the values of our final output node onto a logistic regression curve. If the value post-sigmoid-squish is less than $\frac{1}{2}$, we can change is to $0$, and if it is greater than or equal to $\frac{1}{2}$, we can set it to $1$, thus fulfilling the pre-stated co-domain of $F$.<br>
Editing our forward pass function from above to include $R$ and $\sigma$:
$$\Large F(X)=0 \iff \sigma(W_3(R(W_2(R(W_1X+B_1))+B_2))+B_3) < \frac{1}{2}$$
$$\Large F(X)=1 \iff \sigma(W_3(R(W_2(R(W_1X+B_1))+B_2))+B_3) \ge \frac{1}{2}$$

---

## Loss Function

A loss function is a way for us to measure the correctness of our neural network. Once we get some outputs, we can feed those ouptuts along with ground truths into our loss function to get a quantitative measure of our networks ability. <br>
We will be using Binary Cross-Entropy (BCE) loss, which is defined as follows:
$$\Large \text{J}=-\sum_{i}^{C} y_i\text{log}(\hat{y}_i)$$
With only two classes, the equation simplifies to:
$$\Large \text{J}=-y_1\text{log}(\hat{y}_1)-y_2\text{log}(\hat{y}_2)$$
Or
$$\Large \text{J}=-y_1\text{log}(\hat{y}_1)-(1-y_1)\text{log}(1-\hat{y}_1)$$
Which is the equation we will be using. <br>
<br>
So what does all this mean? 
<br>
Well starting with the new variables we have introduced, 
> J: The output of our loss function, how wrong (or right) is our network?<br>
> $y$: The ground truth value of an input<br>
> $\hat{y}$: The prediction outputted by the forward pass<br>

To be a little more precise with what $\hat{y}$ is, this is the value _before_ setting it to 0 or 1. It is the exact output of the forward pass, thus will be a value between 0 and 1. Because of this, $\hat{y}$ as a _prediction probability_. <br>
To better illustrate this function, let's split it up. We know that the first section of the expression: $y_1\text{log}(\hat{y}_1)$ references the correctness of class 1, whereas $(1-y_1)\text{log}(1-\hat{y}_1)$ references class 2. The reason we represent $y_2$ and $\hat{y}_2$ in terms of $y_1$ and $\hat{y}_1$ will be evident shortly.<br>
Let's now plot the logistic regression curve (sigmoid), and plot some arbitrary points onto it.

<img src=images/math-foundations/logistic-regression.png width=500>

Suppose the ground truth value for each of these points is 1, that is, $y_1=1$. Then, the class 2 term vanishes, leaving just $-y_1\text{log}(\hat{y}_1)$. Well, once again, $y_1=1$ so we only need to worry about $-\text{log}(\hat{y}_1)$.<br>
Looking back at the plot, if the truth label is 1, then the ouput values in blue should be as close to 1 as possible for the most accurate results. It's clear that while $0.94$ and $0.72$ are pretty close, $0.29$ and $0.11$ are not. If these values were fed through the final step of the neural network, $0.94$ and $0.72$ round up to $1$, thus outputting a correct result, but the other two values round down, producing incorrect results. <br>
We can easily determine just how correct each output is, simply by seeing it's distance from 1. $0.94$ is _more_ correct than $0.72$, thus an output of $0.72$ should be penalized heavier than $0.94$, even though they produce the same value. <br>
The penalty is then determined by $-log(x)$. For a nice visualization, let's plot the function first: 

<img src=images/math-foundations/negative-log.png width=500>

It's pretty clear to see that this will serve as a nice penalty function. The closer the value is to 1, the lower the penalty, with values approaching 0 exponentially increasing the penalty.<br>
<br>
But what if the ground truth value is 0? Then $y_1=0$, causing the class 1 term to vanish, leaving just $-(1-y_1)\text{log}(1-\hat{y}_1)$. The same simplifications can occur, thus leaving us with $-\text{log}(1-\hat{y}_1)$. Let's look back at the logistic curve figure to find out what the $(1-\hat{y}_1)$ term refers to:

<img src=images/math-foundations/logistic-regression.png width=500>

If our ground truth is 0, we need to think of the curve but flipped. The closer the value is to 0, the more correct we are. Taking $1-0.11$ yields $0.89$, a number that when you throw into $-\text{log}(x)$, will be penalized much less than say $1-0.94$. By taking $1-\hat{y}_1$, we obtain the correctness if the ground truth is 0, just as we expect. <br>
<br>
In summary, we can define a function J following the formula for Binary Cross-Entropy to obtain a loss function that can quantitatively measure how well our neural network performs. 

---

## Backpropagation

Backpropagation refers to obtaining the gradient of the cost function $J$. The goal is to know how to decrease $J$, based on our learnable parameters, as we will see in the next section dealing with gradient descent. This requires us to determine the effect that each weight and bias has on $J$ through a series of partial derivatives. <br>
Let's simplify this to a case with three nodes:

<img src=/images/math-foundations/three-nodes.png width=500>

Breaking down the new formulas that have been introduced, $J$ is still the cost function, but written explicitly with it's parameters, being the weights and biases. The formula has also been simplified, dropping the subscript for the class and simply taking the $-log$ of the prediction probability. <br>
$z^{(L)}$ is the weighted sum of some layer $L$. Until now, we have just called this $a$, but storing the weighted sum into it's own variable will simplify the derivatives moving forward. <br>
$a^{(L)}$ is still the activation of some node in some layer, however written as a function of $z$, applying some non-linear transformation to $z$ as its output. <br>
<br>
From here, computing the gradient is a classic chain rule exercise. For clarity, the parameters of the cost function $J$ have been explicitly listed, however they would usually be stored in a vector. <br>

$$
\begin{align*}
\Large \frac{\partial J}{\partial w_1}&=\Large \frac{\partial z^{(1)}}{\partial w_1}\frac{\partial a^{(1)}}{\partial z^{(1)}}\frac{\partial z^{(2)}}{\partial a^{(1)}}\frac{\partial a^{(2)}}{\partial z^{(2)}}\frac{\partial J}{\partial a^{(2)}} \\
\Large \frac{\partial J}{\partial b_1}&=\Large \frac{\partial z^{(1)}}{\partial b_1}\frac{\partial a^{(1)}}{\partial z^{(1)}}\frac{\partial z^{(2)}}{\partial a^{(1)}}\frac{\partial a^{(2)}}{\partial z^{(2)}}\frac{\partial J}{\partial a^{(2)}} \\
\Large \frac{\partial J}{\partial w_2}&=\Large \frac{\partial z^{(2)}}{\partial w_2}\frac{\partial a^{(2)}}{\partial z^{(2)}}\frac{\partial J}{\partial a^{(2)}} \\
\Large \frac{\partial J}{\partial b_2}&=\Large \frac{\partial z^{(2)}}{\partial b_2}\frac{\partial a^{(2)}}{\partial z^{(2)}}\frac{\partial J}{\partial a^{(2)}} \\
\end{align*}
$$

To further clarify these expressions, we can set the network up showing it's weights and biases.

<img src=/images/math-foundations/three-nodes-tree.png width=750>

From this figure, computing each partial is just a matter of tracing down the tree to determine the dependence of each variable on $J$.<br>
<br>
Since we defined each step with it's own formula, it's not too hard to determine what the derivatives actually come out to:<br>

$$
\begin{align*}
\Large \frac{\partial z^{(L)}}{\partial w^{(L)}}&=\Large a^{(L-1)} \\
\Large \frac{\partial z^{(L)}}{\partial b^{(L)}}&=\Large 1 \\
\Large \frac{\partial a^{(L)}}{\partial z^{(L)}}&=\Large R'(z^{(L)}) \\
\Large \frac{\partial J}{\partial a^{(\text{final})}}&=\Large -\frac{1}{\sigma({a^{(\text{final})}})}\sigma '(a^{(\text{final})}) \\
\text{or} \\
\Large \frac{\partial J}{\partial a^{(\text{final})}}&=\Large -\frac{1}{\sigma({1-a^{(\text{final})}})}\sigma '(1-a^{(\text{final})}) \\
\end{align*}
$$
