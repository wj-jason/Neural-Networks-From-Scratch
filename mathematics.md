# The Mathematics Behind Neural Networks

This section will cover all the relavent math for a non-linear binary classifier. <br>
It will be broken up into the following sections:
1. Indexing
2. Forward Pass
3. Activation Layers
4. Cost Function
5. Backpropagation
6. Gradient Descent

---

## 1. Indexing

Before diving into the actual math, we need to ensure that there are no ambiguities with respect to what part of the neural net is being referenced.

Starting with the layers themselves, they will be indexing starting from $L=0$. These indices will appear as superscripts surrounded by parenthesis.

For example, in order to reference the inputs, $X$ of layer $0$, it would be notated $X^{(0)}$ (this is a brief allusion to the next section on the forward pass, but $X$ and $Y$ are column vectors holding the inputs and outputs of each neuron in a particular layer). The outputs of layer $0$ are then notated as $Y^{(0)}$, which as we will see soon, act as the next layers inputs. That is, $Y^{(0)}=X^{(1)}$.

INSERT IMAGE HERE

In order to refernce individual activations within the neurons, we will use lowercase $x$ and $y$. These will have both superscipts describing the layer they are in, as well as subscripts denoting the position within the layer. These subscripts will also begin from 0, and are ordered from the top down. 

INSERT IMAGE HERE

Each connection (called a _weight_) will be indexed similar to the neurons. An indiviudal weight $w$ will have a superscript denoting the layer it is _going_ to, with two subscripts denoting the neuron in the next layer it is connected to, followed by the neuron it comes from. Letting the input layer be $n$, and output be $m$, an arbitrary weight is indexed as $w_{mn}^{(L)}$ where $m$ is layer $L$. 

These weights are then stored in a weight matrix $W^{(L)}$ as follows:

$$
\Large W^{(L)}=
\begin{bmatrix}
    w_{00}^{(L)} & w_{01}^{(L)} & w_{02}^{(L)} & \dots  & w_{0n}^{(L)} \\
    w_{10}^{(L)} & w_{11}^{(L)} & w_{12}^{(L)} & \dots  & w_{1n}^{(L)} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    w_{m0}^{(L)} & w_{m1}^{(L)} & w_{m2}^{(L)} & \dots  & w_{mn}^{(L)}
\end{bmatrix}
$$

We can also define a _bias_ term $b$ in the same way as the weight, however only requiring one subscript denoting which neuron it acts on. Then an arbtirary weight in layer $L$ is denoted as $b_n^{(L)}$ with the bias column vector being notated as $B^{(L)}$. 

$$
\Large B^{(L)}=
\begin{bmatrix}
    b_{0}^{(L)} \\
    b_{1}^{(L)}\\
    \vdots \\
    b_{m}^{(L)}
\end{bmatrix}
$$

---

## 2. Forward Pass

The forward pass refernces transformation of the input as it moves throughout the network. Each neuron from the input layer is connected to every single neuron in the output layer. Thus the value of some output neuron $y_m^{(L)}$ is given as a weighted sum of all neurons in the previous layer, plus the bias term for that neuron. 
$$\Large y_m^{(L)}=(w_{m0}^{(L)}x_0^{(L-1)}+w_{m1}^{(L)}x_1^{(L-1)}w_{m2}^{(L)}x_2^{(L-1)}+ ... + w_{mn}^{(L)}x_n^{(L-1)})+b_m^{(L)}$$
Now doing this calculation for every neuron in the output layer wouldn't be too bad, but since we have defined weight matricies and bias vectors, we can store our inputs and outputs in column vectors and take a matrix vector product.
$$\Large Y^{(L)}=W^{(L)}X^{(L-1)}+B^{(L)}$$
Subsctiped with dimensionalities for clarity:
$$\Large Y_{m \text{x} 1}^{(L)}=W_{m \text{x} n}^{(L)}X_{n \text{x} 1}^{(L-1)}+B_{m \text{x} 1}^{(L)}$$
Seeing how this might look for our previous two-layer example network:

INSERT IMAGE HERE

It can now be seen how the outputs for one layer act as the inputs for the next, thus the forward pass through the entire network is only a matter of recursively computing this matrix vector product with each layer. $X$ for one layer produces $Y$ for that layer, which acts as the next $X$ producing another $Y$ and so on. 

---

## 3. Activation Layers

As important as the forward pass matrix vector prodcut is, if that was the _only_ thing done by our network, it would be equivalent to just a fancy linear transformation. Thus if trying to train on non-linear data, the model fails. This is why we introduce non-linear layers into the network. Non-linear layers will have the same number of neurons as it's input, and simply applies a non-linear function to it. Some common examples are as follows:
$$\Large \text{ReLU}(x)=\text{R}(x)=\text{max}(0,x)$$
$$\Large \sigma(x)=\frac{1}{1+e^{-x}}$$
$$\Large \arctan(x)$$
The use is fairly simple. Between each hidden layer (layers between initial input and final output), a non-linear function is added. This function acts on the column vector outputted by the previous layer, applying the non-linear function.

INSERT IMAGE HERE

---

## 4. Cost Function

The cost (also referred to as loss or error) of some layer is the deviation from the expected output. Many functions can serve as cost functions, but we will be using the Mean Squared Error (MSE) function, defined as: 
$$\Large C=\frac{1}{n}\sum_{i=0}^{n-1}(\hat{y}_i-y_i)^2$$
Where $\hat{y}$ is the actual or expected output value.

---

## 5. Backpropagation

Another way to refernce the forward pass is to say _forward propagation_. The given input _propogates_ throughout the network to produce some output.

However as the name suggests, backpropagation is the opposite, and is the process of propogating backwards through the network to calculate the gradients of the cost function. 

Since the cost function is really just a huge function of the _learnable parameters_ (weights and biases), the goal with backpropagation is to find the gradient of the cost with respect to every learnable parameter. From there, we can update the parameteres by some _learning rate_ against the gradient, thus slowly minimizing the cost. 

This proccess is repeated in an iterative manner until the values converge to a local minimum.

As previously stated, the learnable parameters are the weights and biases, so the goal is to find expressions for:
$$\Large \frac{\partial C^{(L)}}{\partial W^{(L)}} \text{  and  } \frac{\partial C^{(L)}}{\partial B^{(L)}}$$

We know that the weights and biases affect the outputs, which in turn affect the cost function. Thus we can use the chain rule to deduce the affects of the weights and biases on the cost.
$$\Large \frac{\partial C^{(L)}}{\partial W^{(L)}}=\frac{\partial Y^{(L)}}{\partial W^{(L)}}\frac{\partial C^{(L)}}{\partial Y^{(L)}}$$
$$\Large \frac{\partial C^{(L)}}{\partial B^{(L)}}=\frac{\partial Y^{(L)}}{\partial B^{(L)}}\frac{\partial C^{(L)}}{\partial Y^{(L)}}$$
Starting with the first expression, we can define $\frac{\partial C^{(L)}}{\partial W^{(L)}}$ as follows:

$$
\Large \frac{\partial C^{(L)}}{\partial W^{(L)}}=
\begin{bmatrix}
    \frac{\partial C^{(L)}}{\partial w_{00}^{(L)}} & \frac{\partial C^{(L)}}{\partial w_{01}^{(L)}} & \frac{\partial C^{(L)}}{\partial w_{02}^{(L)}} & \dots  & \frac{\partial C^{(L)}}{\partial w_{0n}^{(L)}} \\
    \frac{\partial C^{(L)}}{\partial w_{10}^{(L)}} & \frac{\partial C^{(L)}}{\partial w_{11}^{(L)}} & \frac{\partial C^{(L)}}{\partial w_{12}^{(L)}} & \dots  & \frac{\partial C^{(L)}}{\partial w_{1n}^{(L)}} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial C^{(L)}}{\partial w_{m0}^{(L)}} & \frac{\partial C^{(L)}}{\partial w_{m1}^{(L)}} & \frac{\partial C^{(L)}}{\partial w_{m2}^{(L)}} & \dots  & \frac{\partial C^{(L)}}{\partial w_{mn}^{(L)}}
\end{bmatrix}
$$

Computing the first term:

$$\Large \frac{\partial C^{(L)}}{\partial w_{00}^{(L)}}=\Large \frac{\partial Y^{(L)}}{\partial w_{00}^{(L)}}\frac{\partial C^{(L)}}{\partial Y^{(L)}}$$

Which expands to: 
$$\Large \frac{\partial y_0^{(L)}}{\partial w_{00}^{(L)}}\frac{\partial C^{(L)}}{\partial y_0^{(L)}}+\frac{\partial y_1^{(L)}}{\partial w_{00}^{(L)}}\frac{\partial C^{(L)}}{\partial y_1^{(L)}}+...+\frac{\partial y_m^{(L)}}{\partial w_{00}^{(L)}}\frac{\partial C^{(L)}}{\partial y_m^{(L)}}$$

But $w_{00}^{(L)}$ can only affect the neuron it is connected to, $y_0^{(L)}$. Thus we can simplify to:
$$\Large \frac{\partial y_0^{(L)}}{\partial w_{00}^{(L)}}\frac{\partial C^{(L)}}{\partial y_0^{(L)}}$$

The first term can be computed using the forward pass formula:
$$\Large y_0^{(L)}=(w_{00}^{(L)}x_0^{(L-1)}+w_{01}^{(L)}x_1^{(L-1)}+ ... +w_{0n}^{(L)}x_n^{(L-1)})+b_0^{(L)} \implies \frac{\partial Y^{(L)}}{\partial w_{00}^{(L)}}=x_0^{(L-1)}$$

Of course this is not a specfic result:

$$\Large \frac{\partial C^{(L)}}{\partial w_{mn}^{(L)}}=\frac{\partial C^{(L)}}{\partial y_m^{(L)}}x_n^{(L-1)}$$

And so the matrix expands to:

$$
\Large \frac{\partial C^{(L)}}{\partial W^{(L)}}=
\begin{bmatrix}
    \frac{\partial C^{(L)}}{\partial y_0^{(L)}}x_0^{(L-1)} & \frac{\partial C^{(L)}}{\partial y_0^{(L)}}x_1^{(L-1)} & \frac{\partial C^{(L)}}{\partial y_0^{(L)}}x_2^{(L-1)} & \dots  & \frac{\partial C^{(L)}}{\partial y_0^{(L)}}x_n^{(L-1)} \\
    \frac{\partial C^{(L)}}{\partial y_1^{(L)}}x_0^{(L-1)} & \frac{\partial C^{(L)}}{\partial y_1^{(L)}}x_1^{(L-1)} & \frac{\partial C^{(L)}}{\partial y_1^{(L)}}x_2^{(L-1)} & \dots  & \frac{\partial C^{(L)}}{\partial y_1^{(L)}}x_n^{(L-1)} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial C^{(L)}}{\partial y_m^{(L)}}x_0^{(L-1)} & \frac{\partial C^{(L)}}{\partial y_m^{(L)}}x_1^{(L-1)} & \frac{\partial C^{(L)}}{\partial y_m^{(L)}}x_2^{(L-1)} & \dots  & \frac{\partial C^{(L)}}{\partial y_m^{(L)}}x_n^{(L-1)}
\end{bmatrix}
$$

Which is equivalent to the following matrix vector product:
$$\Large \frac{\partial C^{(L)}}{\partial Y^{(L)}}X^{{(L)}^{T}}$$
Dropping the superscripts since everythings in the same layer (and so it looks a little prettier):
$$\boxed{\Huge \frac{\partial C}{\partial W}=\frac{\partial C}{\partial Y}X^T}$$
The proccess for $\frac{\partial C}{\partial B}$ is essentially the exact same. 

Remember $B$ is a column vector with the same dimensionality as $Y$:
$$\Large \frac{\partial C^{(L)}}{\partial b_{0}^{(L)}}=\frac{\partial y_0^{(L)}}{\partial b_{0}^{(L)}}\frac{\partial C^{(L)}}{\partial y_0^{(L)}}+\frac{\partial y_1^{(L)}}{\partial b_{0}^{(L)}}\frac{\partial C^{(L)}}{\partial y_1^{(L)}}+...+\frac{\partial y_m^{(L)}}{\partial b_{0}^{(L)}}\frac{\partial C^{(L)}}{\partial y_m^{(L)}}$$
Which simplifies in the same fashion, as $b_0$ only affects one neuron:
$$\Large \frac{\partial y_0^{(L)}}{\partial b_{0}^{(L)}}\frac{\partial C^{(L)}}{\partial y_0^{(L)}}$$
But referring back to the forward pass formula, every bias is simply a constant, thus it's derivative is $1$:
$$\Large \frac{\partial C^{(L)}}{\partial b_{0}^{(L)}}=\frac{\partial C^{(L)}}{\partial y_0^{(L)}}$$
Generalizing:
$$\Large \frac{\partial C^{(L)}}{\partial b_{m}^{(L)}}=\frac{\partial C^{(L)}}{\partial y_m^{(L)}}$$
Thus the column vector describing the derivatives of each bias on the cost is given simply by:
$$\boxed{\Huge \frac{\partial C}{\partial B}=\frac{\partial C}{\partial Y}}$$


