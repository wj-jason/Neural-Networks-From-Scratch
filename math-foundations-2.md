# Writing the Math for our Network

While math-foundations used specific examples, all the math shown there can be generalized to a neural network of however many layers, nodes, weights, and biases you would like. <br>
So it's easier for us to write everything in code later, let's define every step but specficially for our network. 

---

## Forward Pass + Activation Functions

This won't differ too much from what was in the other document since the example used was for our network. <br>
Just to reiterate though, our forward pass (including activations) is given as: 
$$\Large F(X)=0 \iff \sigma(W_3(R(W_2(R(W_1X+B_1))+B_2))+B_3) < \frac{1}{2}$$
$$\Large F(X)=1 \iff \sigma(W_3(R(W_2(R(W_1X+B_1))+B_2))+B_3) \ge \frac{1}{2}$$
But note that we will need the non-rounded version for our loss function.

---

## Loss Function

The loss function will also stay unchanged, as it is not dependent on the architecture of a network, as long as it is a binary classifier.<br>
$$\Large \text{J}=-y_1\text{log}(\hat{y}_1)-(1-y_1)\text{log}(1-\hat{y}_1)$$
We can also write the shorter version of $J$:
$$\Large \text{J}=-\text{log}(\hat{y})$$

---

## Backpropagation

This is where the definitions get a little more sophisticated, we need to define the cost function derivative with respect to each layers learnable parameters. <br>
In order to better visualize all the formulas, we can use the follwing tree diagram:

<img src=/images/math-foundations-2/tree-diagram.png width=750>

As you can see, this is very similar to the one shown in the previous document, but with an extra layer. Furthermore, instead of each node of the tree representing a specific weight, bias, or activation, each node of the tree represents a vector or matrix encoding the information of each layers weight, bias, and activation. <br>
<br>
First, let's define the formulae for the final layers weights and biases:

$$
\begin{align*}
\Large \frac{\partial J}{\partial W^{(3)}}&=\Large \frac{\partial Z^{(3)}}{\partial W^{(3)}}\frac{\partial A^{(3)}}{\partial Z^{(3)}}\frac{\partial J}{\partial A^{(3)}} \\
\\
&=\boxed{\Large A^{(2)}\sigma '(Z^{(3)})\frac{-1}{A^{(3)}}}
\\
\\
\Large \frac{\partial J}{\partial B^{(3)}}&=\Large \frac{\partial Z^{(3)}}{\partial B^{(3)}}\frac{\partial A^{(3)}}{\partial Z^{(3)}}\frac{\partial J}{\partial A^{(3)}} \\
\\
&=\boxed{\Large \sigma '(Z^{(3)})\frac{-1}{A^{(3)}}}
\end{align*}
$$

The capital letters are being used to denote the relevant values of the _entire_ layer, wth the layer indexed by the superscript. <br>
It's also important to note that $\frac{\partial J}{\partial A^{(3)}}$ has two possible values, either $\frac{-1}{A^{(3)}}$ or $\frac{-1}{1-A^{(3)}}$. The expressions will only show the former, but we will need to program both depending on the class. <br>
<br>
Moving on to layer 2:

$$
\begin{align*}
\Large \frac{\partial J}{\partial W^{(2)}}&=\Large \frac{\partial Z^{(2)}}{\partial W^{(2)}}\frac{\partial A^{(2)}}{\partial Z^{(2)}}\frac{\partial Z^{(3)}}{\partial A^{(2)}}\frac{\partial A^{(3)}}{\partial Z^{(3)}}\frac{\partial J}{\partial A^{(3)}} \\
\\
&=\boxed{\Large A^{(1)}\text{R}'(Z^{(2)})W^{(3)}\sigma '(Z^{(3)})\frac{-1}{A^{(3)}}}
\\
\\
\Large \frac{\partial J}{\partial B^{(2)}}&=\Large \frac{\partial Z^{(2)}}{\partial B^{(2)}}\frac{\partial A^{(2)}}{\partial Z^{(2)}}\frac{\partial Z^{(3)}}{\partial A^{(2)}}\frac{\partial A^{(3)}}{\partial Z^{(3)}}\frac{\partial J}{\partial A^{(3)}} \\
\\
&=\boxed{\Large \text{R}'(Z^{(2)})W^{(3)}\sigma '(Z^{(3)})\frac{-1}{A^{(3)}}}
\end{align*}
$$

And finally, layer 1:

$$
\begin{align*}
\Large \frac{\partial J}{\partial W^{(1)}}&=\Large \frac{\partial Z^{(1)}}{\partial W^{(1)}}\frac{\partial A^{(1)}}{\partial Z^{(1)}}\frac{\partial Z^{(2)}}{\partial A^{(1)}}\frac{\partial A^{(2)}}{\partial Z^{(2)}}\frac{\partial Z^{(3)}}{\partial A^{(2)}}\frac{\partial A^{(3)}}{\partial Z^{(3)}}\frac{\partial J}{\partial A^{(3)}} \\
\\
&=\boxed{\Large \text{X}\text{R}'(Z^{(1)})W^{(2)}\text{R}'(Z^{(2)})W^{(3)}\sigma '(Z^{(3)})\frac{-1}{A^{(3)}}}
\\
\\
\Large \frac{\partial J}{\partial B^{(1)}}&=\Large \frac{\partial Z^{(1)}}{\partial W^{(1)}}\frac{\partial A^{(1)}}{\partial Z^{(1)}}\frac{\partial Z^{(2)}}{\partial A^{(1)}}\frac{\partial A^{(2)}}{\partial Z^{(2)}}\frac{\partial Z^{(3)}}{\partial A^{(2)}}\frac{\partial A^{(3)}}{\partial Z^{(3)}}\frac{\partial J}{\partial A^{(3)}} \\
\\
&=\boxed{\Large \text{R}'(Z^{(1)})W^{(2)}\text{R}'(Z^{(2)})W^{(3)}\sigma '(Z^{(3)})\frac{-1}{A^{(3)}}}
\end{align*}
$$
