# The Mathematics Behind Neural Networks

1. [Indexing](#1-indexing)
2. [Forward Pass](#2-forward-pass)
3. [Activation Layers](#3-activation-layers)
4. [Cost Function](#4-cost-function)
5. [Backpropagation 1](#5-backpropagation-1)
6. [Backpropagation 2](#6-backpropagation-2)
7. [Gradient Descent](#7-gradient-descent)
8. [Example Network](#8-example-network)

---

## 1. Indexing

Before diving into the actual math, we need to ensure that there are no ambiguities with respect to what part of the neural net is being referenced.

Starting with the layers themselves, they will be indexing starting from $L=0$. These indices will appear as superscripts surrounded by parenthesis.

For example, in order to reference the inputs, $X$ of layer $0$, it would be notated $X^{(0)}$ (this is a brief allusion to the next section on the forward pass, but $X$ and $Y$ are column vectors holding the inputs and outputs of each neuron in a particular layer). The outputs of layer $0$ are then notated as $Y^{(0)}$, which as we will see soon, act as the next layers inputs. That is, $Y^{(0)}=X^{(1)}$.

<img src=/images/mathematics/fig_0.png width=500>

In order to refernce individual activations within the neurons, we will use lowercase $x$ and $y$. These will have both superscipts describing the layer they are in, as well as subscripts denoting the position within the layer. These subscripts will also begin from 0, and are ordered from the top down. 

<img src=/images/mathematics/fig_1.png width=500>

Each connection (called a _weight_) will be indexed similar to the neurons. An indiviudal weight $w$ will have a superscript denoting the layer it's in with two subscripts denoting the neuron it is connected to, followed by the neuron it comes from. Letting the input neuron be $n$, and output be $m$, an arbitrary weight is indexed as $w_{mn}^{(L)}$.

<img src=/images/mathematics/fig_2.png width=500>

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

While it's important to take note of the superscripts for layers, we will be dropping them from now on as all computations will be done within one layers reach thus all superscripts would be the same.

---

## 2. Forward Pass

The forward pass refernces transformation of the input as it moves throughout the network. Each neuron from the input layer is connected to every single neuron in the output layer. Thus the value of some output neuron $y_m$ is given as a weighted sum of all neurons in the previous layer, plus the bias term for that neuron. 
$$\Large y_m=(w_{m0}x_0+w_{m1}x_1w_{m2}x_2+ ... + w_{mn}x_n)+b_m$$
Now doing this calculation for every neuron in the output layer wouldn't be too bad, but since we have defined weight matricies and bias vectors, we can store our inputs and outputs in column vectors and take a matrix vector product.
$$\Large Y=WX+B$$
Subsctiped with dimensionalities for clarity:
$$\Large Y_{m \text{x} 1}=W_{m \text{x} n}X_{n \text{x} 1}+B_{m \text{x} 1}$$
Seeing how this might look for our previous example network:

<img src=/images/mathematics/fig_3.png width=500>

It can now be seen how the outputs for one layer act as the inputs for the next, thus the forward pass through the entire network is only a matter of recursively computing this matrix vector product with each layer. $X$ for one layer produces $Y$ for that layer, which acts as the next $X$ producing another $Y$ and so on. 

<img src=/images/mathematics/fig_4.png width=500>

---

## 3. Activation Layers

As important as the forward pass matrix vector prodcut is, if that was the _only_ thing done by our network, it would be equivalent to just a fancy linear transformation. Thus if trying to train on non-linear data, the model fails. This is why we introduce non-linear layers into the network. Non-linear layers will have the same number of neurons as it's input, and simply applies a non-linear function to it. Some common examples are as follows:
$$\Large \text{ReLU}(x)=\text{R}(x)=\text{max}(0,x)$$
$$\Large \sigma(x)=\frac{1}{1+e^{-x}}$$
$$\Large \arctan(x)$$
The use is fairly simple. Between each hidden layer (layers between initial input and final output), a non-linear function is added. This function acts on the column vector outputted by the previous layer, applying the non-linear function.

<img src=/images/mathematics/fig_5.png width=400> 

---

## 4. Cost Function

The cost (also referred to as loss or error) of some layer is the deviation from the expected output. Many functions can serve as cost functions, but we will be using the Mean Squared Error (MSE) function, defined as: 
$$\Large C=\frac{1}{n}\sum_{i=0}^{n-1}(\hat{y}_i-y_i)^2$$
Where $\hat{y}$ is the actual or expected output value.

---

## 5. Backpropagation 1

Another way to reference the forward pass is to say _forward propagation_. The given input _propogates_ throughout the network to produce some output.

However as the name suggests, backpropagation is the opposite, and is the process of propogating backwards through the network to calculate the gradients of the cost function. 

Since the cost function is really just a huge function of the _learnable parameters_ (weights and biases), the goal with backpropagation is to find the gradient of the cost with respect to every learnable parameter. From there, we can update the parameteres by some _learning rate_ against the gradient, thus slowly minimizing the cost. 

This proccess is repeated in an iterative manner until the values converge to a local minimum.

As previously stated, the learnable parameters are the weights and biases, so the goal is to find expressions for:
$$\Large \frac{\partial C}{\partial W} \text{  and  } \frac{\partial C}{\partial B}$$

We know that the weights and biases affect the outputs, which in turn affect the cost function. Thus we can use the chain rule to deduce the affects of the weights and biases on the cost.
$$\Large \frac{\partial C}{\partial W}=\frac{\partial Y}{\partial W}\frac{\partial C}{\partial Y}$$
$$\Large \frac{\partial C}{\partial B}=\frac{\partial Y}{\partial B}\frac{\partial C}{\partial Y}$$
Starting with the first expression, we can define $\frac{\partial C}{\partial W}$ as follows:

$$
\Large \frac{\partial C}{\partial W}=
\begin{bmatrix}
    \frac{\partial C}{\partial w_{00}} & \frac{\partial C}{\partial w_{01}} & \frac{\partial C}{\partial w_{02}} & \dots  & \frac{\partial C}{\partial w_{0n}} \\
    \frac{\partial C}{\partial w_{10}} & \frac{\partial C}{\partial w_{11}} & \frac{\partial C}{\partial w_{12}} & \dots  & \frac{\partial C}{\partial w_{1n}} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial C}{\partial w_{m0}} & \frac{\partial C}{\partial w_{m1}} & \frac{\partial C}{\partial w_{m2}} & \dots  & \frac{\partial C}{\partial w_{mn}}
\end{bmatrix}
$$

Computing the first term:

$$\Large \frac{\partial C}{\partial w_{00}}=\Large \frac{\partial Y}{\partial w_{00}}\frac{\partial C}{\partial Y}$$

Which expands to: 
$$\Large \frac{\partial y_0}{\partial w_{00}}\frac{\partial C}{\partial y_0}+\frac{\partial y_1}{\partial w_{00}}\frac{\partial C}{\partial y_1}+...+\frac{\partial y_m}{\partial w_{00}}\frac{\partial C}{\partial y_m}$$

But $w_{00}$ can only affect the neuron it is connected to, $y_0$. Thus we can simplify to:
$$\Large \frac{\partial y_0}{\partial w_{00}}\frac{\partial C}{\partial y_0}$$

The first term can be computed using the forward pass formula:
$$\Large y_0=(w_{00}x_0+w_{01}x_1+ ... +w_{0n}x_n)+b_0 \implies \frac{\partial Y}{\partial w_{00}}=x_0$$

Of course this is not a specfic result:

$$\Large \frac{\partial C}{\partial w_{mn}}=\frac{\partial C}{\partial y_m}x_n$$

And so the matrix expands to:

$$
\Large \frac{\partial C}{\partial W}=
\begin{bmatrix}
    \frac{\partial C}{\partial y_0}x_0 & \frac{\partial C}{\partial y_0}x_1 & \frac{\partial C}{\partial y_0}x_2 & \dots  & \frac{\partial C}{\partial y_0}x_n \\
    \frac{\partial C}{\partial y_1}x_0 & \frac{\partial C}{\partial y_1}x_1 & \frac{\partial C}{\partial y_1}x_2 & \dots  & \frac{\partial C}{\partial y_1}x_n \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial C}{\partial y_m}x_0 & \frac{\partial C}{\partial y_m}x_1 & \frac{\partial C}{\partial y_m}x_2 & \dots  & \frac{\partial C}{\partial y_m}x_n
\end{bmatrix}
$$

Which can be written as a matrix vector product, yeilding the following result:
$$\boxed{\Huge \frac{\partial C}{\partial W}=\frac{\partial C}{\partial Y}X^T}$$

The proccess for $\frac{\partial C}{\partial B}$ is essentially the exact same. 

The target result is a column vector as such:

$$
\Large \frac{\partial C}{\partial B}=
\begin{bmatrix}
    \frac{\partial C}{\partial b_0} \\
    \frac{\partial C}{\partial b_1} \\
    \vdots \\
    \frac{\partial C}{\partial b_m} \\
\end{bmatrix}
$$

Following a similar process to the weights:

$$\Large \frac{\partial C}{\partial b_{0}}=\frac{\partial y_0}{\partial b_{0}}\frac{\partial C}{\partial y_0}+\frac{\partial y_1}{\partial b_{0}}\frac{\partial C}{\partial y_1}+...+\frac{\partial y_m}{\partial b_{0}}\frac{\partial C}{\partial y_m}$$

Which simplifies in the same fashion, as $b_0$ only affects one neuron:
$$\Large \frac{\partial y_0}{\partial b_{0}}\frac{\partial C}{\partial y_0}$$

But referring back to the forward pass formula, every bias is simply a constant, thus it's derivative is $1$:
$$\Large \frac{\partial C}{\partial b_{0}}=\frac{\partial C}{\partial y_0}$$

Generalizing:
$$\Large \frac{\partial C}{\partial b_{m}}=\frac{\partial C}{\partial y_m}$$

Thus the vector is now:

$$
\Large \frac{\partial C}{\partial B}=
\begin{bmatrix}
    \frac{\partial C}{\partial y_0} \\
    \frac{\partial C}{\partial y_1} \\
    \vdots \\
    \frac{\partial C}{\partial y_m} \\
\end{bmatrix}
$$

In other words:
$$\boxed{\Huge \frac{\partial C}{\partial B}=\frac{\partial C}{\partial Y}}$$

Finally, we also need to compute $\frac{\partial C}{\partial X}$.

The reason why may not be obvious now, but we cannot compute $\frac{\partial C}{\partial Y}$ without it. The details to this will be explored in the third backpropagation section, but to summarize, in order to compute the derivative of cost with respect to the output (as we do for the weights and biases), we can use the inputs of the next layer as the inputs of layer $L+1$ are the outputs of layer $L$. 

Just as the biases, the target is a column vector:

$$
\Large \frac{\partial C}{\partial X}=
\begin{bmatrix}
    \frac{\partial C}{\partial x_0} \\
    \frac{\partial C}{\partial x_1} \\
    \vdots \\
    \frac{\partial C}{\partial x_n} \\
\end{bmatrix}
$$

Some input $x_n$ affects the entirety of the output vector $Y$. Thus we get the following:
$$\Large \frac{\partial C}{\partial x_n}=\frac{\partial y_0}{\partial x_n}\frac{\partial C}{\partial y_0}+\frac{\partial y_1}{\partial x_n}\frac{\partial C}{\partial y_1}+...+\frac{\partial y_m}{\partial x_n}\frac{\partial C}{\partial y_m}$$

Now unlike the weights and biases, the input neuron affects every output neuron. 

Once again, referring back to the forward pass, $x_n$ appears with coefficeint $w_{mn}$ for all $m$. 

Thus the derivative works out as:
$$\Large \frac{\partial C}{\partial x_n}=\frac{\partial C}{\partial y_0}w_{0n}+\frac{\partial C}{\partial y_1}w_{1n}+...+\frac{\partial C}{\partial y_m}w_{mn}$$

Applying the formula we have just derived to the target vector:

$$
\Large \frac{\partial C}{\partial B}=
\begin{bmatrix}
    \frac{\partial C}{\partial y_0}w_{00}+\frac{\partial C}{\partial y_1}w_{10}+...+\frac{\partial C}{\partial y_m}w_{m0} \\
    \frac{\partial C}{\partial y_0}w_{01}+\frac{\partial C}{\partial y_1}w_{11}+...+\frac{\partial C}{\partial y_m}w_{m1} \\
    \vdots \\
    \frac{\partial C}{\partial y_0}w_{0n}+\frac{\partial C}{\partial y_1}w_{1n}+...+\frac{\partial C}{\partial y_m}w_{mn} \\
\end{bmatrix}
$$

Which once again, is a matrix vector product, thus the final result is:
$$\boxed{\Huge \frac{\partial C}{\partial X}=W^T\frac{\partial C}{\partial Y}}$$

To finally conclude this section, let's look at the three expressions we have derived:

$$
\begin{align*}
\Huge \frac{\partial C}{\partial W}&=\Huge \frac{\partial C}{\partial Y}X^T \\
\\
\Huge \frac{\partial C}{\partial B}&=\Huge \frac{\partial C}{\partial Y} \\
\\
\Huge \frac{\partial C}{\partial X}&=\Huge W^T\frac{\partial C}{\partial Y}
\end{align*}
$$

---

## 6. Backpropagation 2

So now we have our formulas for backpropagation on the forward pass, but what about the activation layers?

Luckily this is much simpler, since all the activation layer does is pass the input through a pre-defined non-linear function. 

Given a set of inputs $X$ and outputs $Y$, the activation layer applies a function $f$ element-wise to each component of $X$. That is, $Y=f(X)$

While there are no parameters to tune, we still need the derivative of the the cost function with respect to the input. 

Some element $x_n \in X$ undergoes transformation by function $f$ as previously stated. This then affects the corresponding output $y_n \in Y$, thus we can define the derivative of the cost with respect to some $x_n$ as:
$$\Large \frac{\partial C}{\partial x_n}=\frac{\partial y_n}{\partial x_n}\frac{\partial C}{\partial y_n}$$

The first term is simply the derivative of the activation function $f$. Thus generalizing and simplifying as in the previous section:
$$\boxed{\Huge \frac{\partial C}{\partial X}=\frac{\partial C}{\partial Y} \odot f'(X)}$$

Where $\odot$ represents element-wise multiplication

---

## 7. Gradient Descent

Now that we have all the derivatives, we can begin the "learning" process.

There term "learning" refers to the minimization of the cost function by adjusting our learnable parameters against the gradients, which we have computed.

The formula for gradient descent for some matrix / vector of learnable parameters $\theta$ is given as:
$$\Large \theta:=\theta-\alpha\frac{\partial C}{\partial \theta}$$

Where $\alpha$ is the _learning rate_, or in other words the amount the parameters are adjusted by. 

---

## 8. Example Network

We now have all the tools required to set up a basic neural network! For this example, we will keep it extremely simplified, however all the concepts generalize without issues. 

The network we will be looking at will have the follwing architecture:

<img src=/images/mathematics/fig_6.png width=500>

Where layer $1$ is an activation layer, made evident by the single connections.

We can observe the inputs and outputs through the forward pass as follows:

<img src=/images/mathematics/fig_7.png width=500>

And we can see which values we need to compute on backpropagation:

<img src=/images/mathematics/fig_8.png width=500>

Note there are no extra computations on $\frac{\partial C}{\partial Y^{(1)}}$ since it is an activation layer, and has no learnable parameters.

Now starting the computations, the output $Y^{(2)}$ can be used to begin the backpropagation cycle:

$$
\begin{align*}
\Large \frac{\partial C}{\partial Y^{(2)}}&=\Large 2(Y-\hat{Y}) \\
\\
\Large \implies \frac{\partial C}{\partial W^{(2)}}&=\Large 2(Y-\hat{Y})X^{{(2)}^T} \\
\\
\Large \implies \frac{\partial C}{\partial B^{(2)}}&=\Large 2(Y-\hat{Y}) \\
\\
\Large \implies \frac{\partial C}{\partial X^{(2)}}&=\Large W^{{(2)}^T}2(Y-\hat{Y})
\end{align*}
$$

We know $\frac{\partial C}{\partial X^{(2)}}=\frac{\partial C}{\partial Y^{(1)}}$, so:

$$
\begin{align*}
\Large \frac{\partial C}{\partial Y^{(1)}}&=\Large W^{{(2)}^T}2(Y-\hat{Y}) \\
\\
\Large \implies \frac{\partial C}{\partial X^{(1)}}&=\Large W^{{(2)}^T}2(Y-\hat{Y}) \odot \sigma '(X^{(1)})
\end{align*}
$$

Once again, $\frac{\partial C}{\partial X^{(1)}}=\frac{\partial C}{\partial Y^{(0)}}$, so:

$$
\begin{align*}
\Large \frac{\partial C}{\partial Y^{(0)}}&=\Large W^{{(2)}^T}2(Y-\hat{Y}) \odot \sigma '(X^{(1)}) \\
\\
\Large \implies \frac{\partial C}{\partial W^{(0)}}&=\Large W^{{(2)}^T}2(Y-\hat{Y}) \odot \sigma '(X^{(1)})X^{{(0)}^T} \\
\\
\Large \implies \frac{\partial C}{\partial B^{(0)}}&=\Large W^{{(2)}^T}2(Y-\hat{Y}) \odot \sigma '(X^{(1)})
\end{align*}
$$

There is no need to compute $\frac{\partial C}{\partial X^{(0)}}$, so we are done!

From here, we can update our parameters as follows:

$$
\begin{align*}
\Large W^{(2)}&:=\Large W^{(2)}-\alpha\frac{\partial C}{\partial W^{(2)}}\\
\\
\Large &:=\Large W^{(2)}-\Large \alpha \cdot 2(Y-\hat{Y})X^{{(2)}^T} \\
\\
\\
\Large B^{(2)}&:=\Large B^{(2)}-\alpha \frac{\partial C}{\partial B^{(2)}}\\
\\
\Large &:=\Large W^{(2)}-\Large \alpha \cdot2(Y-\hat{Y}) \\
\\
\\
\Large W^{(0)}&:=\Large W^{(0)}-\alpha\frac{\partial C}{\partial W^{(0)}}\\
\\
\Large &:=\Large W^{(0)}-\Large \alpha \cdot W^{{(2)}^T}2(Y-\hat{Y}) \odot \sigma '(X^{(1)})X^{{(0)}^T} \\
\\
\\
\Large B^{(0)}&:=\Large B^{(0)}-\alpha\frac{\partial C}{\partial B^{(0)}}\\
\\
\Large &:=\Large B^{(0)}-\Large \alpha \cdot 2(Y-\hat{Y}) \odot \sigma '(X^{(1)})X^{{(0)}^T} \\
\end{align*}
$$

And that is the complete learning process for a neural network!
