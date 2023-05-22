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

<img src=/images/math-foundations-2/tree-diagram.png width=500>
