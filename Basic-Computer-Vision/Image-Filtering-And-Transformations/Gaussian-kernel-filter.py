"""

Gaussian - Filter

"""

def plot_cv_img(input_image, output_image):
 """
 Converts an image from BGR to RGB and plots
 """
 fig, ax = plt.subplots(nrows=1, ncols=2)

 ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
 ax[0].set_title('Input Image')
 ax[0].axis('off')

 ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
 ax[1].set_title('Gaussian Blurred')
 ax[1].axis('off')
 plt.show()

def main():
 # read an image
 img = cv2.imread('../figures/flower.png')

 # apply gaussian blur,

 # kernel of size 5x5,
 # change here for other sizes
 kernel_size = (5,5)
 # sigma values are same in both direction
 blur = cv2.GaussianBlur(img,(5,5),0)

 plot_cv_img(img, blur)
if __name__ == '__main__':
 main()

