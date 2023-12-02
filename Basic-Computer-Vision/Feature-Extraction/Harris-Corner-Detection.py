# We begin with choosing a matrix, termed a window, which is small in size as compared to the image size.
# The basic idea is to first overlay chosen window on the input image and observe only the overlayed region from the input image. 
# This window is later shifted over the image and the new overlayed region is observed.

"""
Case -1 

If there is a flat surface, then we won't be able to see any change in the window region
irrespective of the direction of movement of the window. This is because there is no
edge or corner in the window region.

"""

"""
Case - 2

In our second case, the window is overlayed on edge in the image and shifted. If the
window moves along the direction of the edge, we will not be able to see any changes
in the window. While, if the window is moved in any other direction, we can easily
observe changes in the window region.

"""

"""
Case - 3 

In our second case, the window is overlayed on edge in the image and shifted. If the
window moves along the direction of the edge, we will not be able to see any changes
in the window. While, if the window is moved in any other direction, we can easily
observe changes in the window region.

"""

# The Harris Corner Detection score value will show whether there is an edge, corner, or flat surface.

# load image and convert to grayscale
img = cv2.imread('../figures/flower.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# harris corner parameters
block_size = 4 # Covariance matrix size
kernel_size = 3 # neighbourhood kernel
k = 0.01 # parameter for harris corner score
# compute harris corner
corners = cv2.cornerHarris(gray, block_size, kernel_size, k)
# create corner image
display_corner = np.ones(gray.shape[:2])
display_corner = 255*display_corner
# apply thresholding to the corner score
thres = 0.01 # more than 1% of max value
display_corner[corners>thres*corners.max()] = 10 #display pixel value
# set up display
plt.figure(figsize=(12,8))
plt.imshow(display_corner, cmap='gray')
plt.axis('off')


"""

We can generate different number of corners for an image by changing the parameters such
as covariance matrix block size, neighbourhood kernel size and Harris score parameter.

"""