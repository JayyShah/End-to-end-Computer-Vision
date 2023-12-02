# downsample image by halving both width and height
# input:(h, w) --> output:(h/2, w/2)
lower_resolution_img = cv2.pyrDown(img)

# Upsamples image by doubling both width and height
# input:(h, w) --> output:(h*2, w*2)
higher_resolution_img = cv2.pyrUp(img)
