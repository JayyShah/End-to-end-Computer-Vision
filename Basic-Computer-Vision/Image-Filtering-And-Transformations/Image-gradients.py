"""

Image-Gradients 

"""

"""
Sobel

"""
x_sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
y_sobel = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

"""
Laplacian

"""
lapl = cv2.Laplacian(img,cv2.CV_64F, ksize=5)


"""
Gaussian-Blur

"""

blur = cv2.GaussianBlur(img,(5,5),0)
# laplacian of gaussian
log = cv2.Laplacian(blur,cv2.CV_64F, ksize=5)