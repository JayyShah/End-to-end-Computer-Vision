def create_model(img_h=28, img_w=28):
 """
 Creates a CNN model for training.
 Inputs:
 img_h: input image height
 img_w: input image width
 Returns:
 Model structure
 """

 inputs = Input(shape=(img_h, img_w, 1))
 x = conv3x3(inputs, 32)
 x = conv3x3(x, 32)
 x = MaxPooling2D(pool_size=(2,2))(x)
 x = conv3x3(x, 64)
 x = conv3x3(x, 64)
 x = MaxPooling2D(pool_size=(2,2))(x)
 x = conv3x3(x, 128)
 x = MaxPooling2D(pool_size=(2,2))(x)
 x = Flatten()(x)
 x = Dense(128, activation="relu")(x)
 preds = Dense(nb_class, activation='softmax')(x)

 model = Model(inputs=inputs, outputs=preds)
 print(model.summary())
 return model