"""

Deep learning based segmentation approaches have recently grown, both in terms of
accuracy as well as effectiveness, in more complex domains. One of the popular models
using CNN for segmentation is a fully convolutional network (FCN)

-----------------------------------------------------------------------------------------------------------------------------

This method has the advantage of training an end-to-end CNN to perform pixel-wise semantic segmentation. 
The output is an image with each pixel classified as either background or into one of the predefined categories of objects.

-----------------------------------------------------------------------------------------------------------------------------

As the layers are stacked hierarchically, the output from each layer gets downsampled yet is
feature rich. In the last layer, the downsampled output is upsampled using a deconvolutional layer,
resulting in the final output being the same size as that of the input.

The deconvolutional layer is used to transform the input feature to the upsampled feature,
however, the name is a bit misleading, as the operation is not exactly the inverse of
convolution. This acts as transposed convolution, where the input is convolved after a
transpose, as compared to a regular convolution operation.

--------------------------------------------------------------------------------------------------------------------------------

The feature extractor is kept the same, while upsampling is updated with more deconvolutional layers where each of these layers 
upsamples features from the previous layer and generates an overall richer prediction.

"""

# Modules

from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16



def create_model_fcn32(nb_class, input_w=256):
 """
 Create FCN-32s model for segmentaiton.
 Input:
 nb_class: number of detection categories
 input_w: input width, using square image
 Returns model created for training.
 """
 input = Input(shape=(input_w, input_w, 3))
 # initialize feature extractor excuding fully connected layers
 # here we use VGG model, with pre-trained weights.
 vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input)
 # create further network
 x = Conv2D(4096, kernel_size=(7,7), use_bias=False,
 activation='relu', padding="same")(vgg.output)
 x = Dropout(0.5)(x)
 x = Conv2D(4096, kernel_size=(1,1), use_bias=False,
 activation='relu', padding="same")(x)
 x = Dropout(0.5)(x)
 x = Conv2D(nb_class, kernel_size=(1,1), use_bias=False,
 padding="same")(x)
 # upsampling to image size using transposed convolution layer
 x = Conv2DTranspose(nb_class ,
 kernel_size=(64,64),
strides=(32,32),
 use_bias=False, padding='same')(x)
 x = Activation('softmax')(x)
 model = Model(input, x)
 model.summary()
 return model
# Create model for pascal voc image segmentation for 21 classes
model = create_model_fcn32(21)

# The upsampling method is key to compute pixel-wise categories and hence different choices of
# upsampling methods will result in a different quality of results.
