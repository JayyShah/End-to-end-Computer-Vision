# In the Previous Code, used a convolution layer and added a pooling operation to it


from keras.layers import Conv2D, Input, MaxPooling2D
from keras.models import Model
def print_model():
 """
 Creates a sample model and prints output shape
 Use this to analyse Pooling parameters
 """
 # create input with given shape
 x = Input(shape=(512,512,3))
 # create a convolution layer
 conv = Conv2D(filters=32,
 kernel_size=(5,5), activation="relu",
 strides=1, padding="same",
 use_bias=True)(x)
 pool = MaxPooling2D(pool_size=(2,2))(conv)

 # create model
 model = Model(inputs=x, outputs=pool)
 # prints our model created
 model.summary()
print_model()


"""

The pooling parameter to (2,2), representing the width and height of the pooling operation. 
The depth for pooling will be set according to the depth of the input to the pooling layer. 
The resulting output is of half the shape in terms of width and height; however, there is no change in depth size.

"""