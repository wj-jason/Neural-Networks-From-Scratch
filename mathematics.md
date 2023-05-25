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

Before diving into the actual math, we need to ensure that there are no ambiguities with respect to what part of the neural net is being referenced. <br>
Starting with the layers themselves, they will be indexing starting from $L=0$. These indices will appear as superscripts surrounded by parenthesis. <br>
For example, in order to reference the inputs, $X$ of layer $0$, it would be notated $X^{(0)}$. The outputs of layer $0$ are then notated as $Y^{(0)}$, which as we will see soon, act as the next layers inputs. That is, $Y^{(0)}=X^{(1)}$.
