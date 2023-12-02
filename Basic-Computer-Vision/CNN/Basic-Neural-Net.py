import numpy as np

dim_x = 1000 # input dims
dim_y = 2 # output dims
batch = 10 # batch size for training
lr = 1e-4 # learning rate for weight update
steps = 5000 # steps for learning
# create random input and targets
x = np.random.randn(batch, dim_x)
y = np.random.randn(batch, dim_y)
# initialize weight matrix
w = np.random.randn(dim_x, dim_y)


def net(x, w):
 """
 A simple neural net that performs non-linear transformation
 Function : 1 / (1 + e^(-w*x))
 x: inputs
 w: weight matrix
 Returns the function value
 """
 return 1/(1+np.exp(-x.dot(w)))

"""

To learn these weights w, we will follow the gradient descent method. For each input we will
compute the gradient of loss with respect to w and update the weights as follows:

"""

"""

This step is iterated repeatedly over our labeled dataset until our loss does not change
significantly or the loss values start following some repetition. The loss function is defined
as,

"""



def compute_loss(y, y_pred):
 """
 Loss function : sum(y_pred**2 - y**2)
 y: ground truth targets
 y_pred: predicted target values
 """
 return np.mean((y_pred-y)**2)


def backprop(y, y_pred, w, x):
 """
 Backpropagation to compute gradients of weights
 y : ground truth targets
 y_pred : predicted targets
 w : weights for the network
 x : inputs to the net
 """
 # start from outer most
 y_grad = 2.0 * (y_pred - y)
 # inner layer grads
 w_grad = x.T.dot(y_grad * y_pred * (1 - y_pred))
 return w_grad
for i in range(steps):
 # feed forward pass
 y_pred = net(x, w)
 # compute loss
 loss = compute_loss(y, y_pred)
 print("Loss:", loss, "at step:", i)
 # compute grads using backprop on given net
 w_grad = backprop(y, y_pred, w, x)
 # update weights with some learning rate
 w -= lr * w_grad

 """

We can see the values of loss decreasing and settling down.The parameters here are learning rate and initial w values. A good choice for these values
may cause the loss to decrease faster and settle early; however, a bad choice will lead to no decrease in loss, or sometimes an increase in loss over several iterations.

"""

"""

You can use this code and modify or add complex structures to play with it.

"""

