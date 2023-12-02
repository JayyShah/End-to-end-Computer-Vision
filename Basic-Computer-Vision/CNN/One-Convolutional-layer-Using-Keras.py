from keras.layers import Conv2D, Input
from keras.models import Model
def print_model():
 """
 Creates a sample model and prints output shape
 Use this to analyse convolution parameters
 """
 # create input with given shape
 x = Input(shape=(512,512,3))
 # create a convolution layer
 y = Conv2D(filters=32,
 kernel_size=(5,5),
 strides=1, padding="same",
 use_bias=False)(x)

 # create model
 model = Model(inputs=x, outputs=y)
 # prints our model created
 model.summary()
print_model()


# KEY POINTS

"""

We have set the input to be of the shape 512 x 512 x 3, and for convolution, we use 32 filters with the size 5 x 5. 
The stride values we have set to 1, and using the same padding for the edges, we make sure a kernel captures all of the images.

The total number of parameters for this layer is 5 x 5 x 3 x 32 (kernel_size * number of filters) which is 2400.


----------------------------------------------------------------------------------------------------------

# Try Another Run; Set the Stride to 2 on the above code and check the Output.



The Convolution Output Shape (Height, Width) is reduced to Half of the input size. This is due to the stride
option chosen. Using strides 2, it will skip one pixel, making the Output half of the input.


-------------------------------------------------------------------------------------------------------------

Setting Stride = 1 and Padding = valid, the Output shape (width and height) is reduced to 508.This is due to lack 
of padding set and the kernel cannot be applied to the edges of the input.

Output Shape can be computed as (I-K+2P)/S+1 ;
I = Input Size , K= Kernel Size , P is Padding used and S is the Stride Value.

If we use same padding, the P value is (K-1)/2 . Else if Valid padding is used then P value is 0.


---------------------------------------------------------------------------------------------------------------

Set another parameter use_bias=False. On setting this as true, it will add a constant value to each kernel and for a convolution layer, 
the bias parameter is the same as the number of filters used.

"""