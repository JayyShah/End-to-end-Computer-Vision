"""
The most common choice of activation function is Rectified Linear Unit (ReLU) and this
performs well in the majority of cases. In Convolutional Layer code activation layer can be added as:
"""


from keras.layers import Conv2D, Input, Activation
from keras.models import Model
def print_model():
 """
 Creates a sample model and prints output shape
 Use this to analyse convolution parameters
 """
 # create input with given shape
 x = Input(shape=(512,512,3))
 # create a convolution layer
 conv = Conv2D(filters=32,
 kernel_size=(5,5),
 strides=1, padding="same",
 use_bias=True)(x)

 # add activation layer
 y = Activation('relu')(conv)

 # create model
 model = Model(inputs=x, outputs=y)
 # prints our model created
 model.summary()
print_model()


# In Keras, the activation layer can also be added to the convolution layer as 

"""
conv = Conv2D(filters=32,
kernel_size=(5,5), activation="relu"
strides=1, padding="same",
use_bias=True)(x)

"""