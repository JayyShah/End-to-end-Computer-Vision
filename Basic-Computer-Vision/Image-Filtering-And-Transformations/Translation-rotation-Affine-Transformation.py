"""
Translation

"""


# input shape
w, h = flower.shape[1], flower.shape[0]
# create translation matrix
tx = w/2 # half of width
ty = h/2 # half of height
translation_matrix = np.float32([[1,0,tx],
 [0,1,ty]])
# apply translation operation using warp affine function.
output_size = (w*2,h*2)
translated_flower = cv2.warpAffine(flower, translation_matrix, output_size)


"""

Rotation

"""

# input shape
w, h = flower.shape[1], flower.shape[0]
# create rotation matrix
rot_angle = 90 # in degrees
scale = 1 # keep the size same
rotation_matrix = cv2.getRotationMatrix2D((w/2,h/2),rot_angle,1)
# apply rotation using warpAffine
output_size = (w*2,h*2)
rotated_flower = cv2.warpAffine(flower,rotation_matrix,output_size)


"""
Affine - Transformation

"""

# create transformation matrix form preselected points
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
affine_tr = cv2.getAffineTransform(pts1,pts2)
transformed = cv2.warpAffine(img, affine_tr, (img.shape[1]*2,img.shape[0]*2))
