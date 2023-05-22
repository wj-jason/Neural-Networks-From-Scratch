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
