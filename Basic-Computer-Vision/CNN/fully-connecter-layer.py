"""

This is a simple neural network layer where each neuron in the current layer is connected to
all the neurons in the previous layer. This is often referred to as Dense or Linear in various
deep learning libraries.

"""

from keras.layers import Dense, Input
from keras.models import Model
def print_model():
 """
 Creates a sample model and prints output shape
 Use this to analyse dense/Fully Connected parameters
 """
 # create input with given shape
 x = Input(shape=(512,))
 # create a fully connected layer layer
 y = Dense(32)(x)

 # create model
 model = Model(inputs=x, outputs=y)
 # prints our model created
 model.summary()
print_model()


"""

* KEY POINTS

The total parameters for this layer are given by (Is*Os) + Os
Where, Is = Input shape, Os = Output Shape. 

In the above code we used an input shape of 512 and an output shape of 32 and get a total parameters with bias.0

This is quite large compared to a similar convolution layer block, therefore in recent models, there has been a trend towards
using more convolution blocks rather than fully connected blocks.

Nonetheless, this layer
still plays a major role in designing simple convolution neural net blocks.

"""