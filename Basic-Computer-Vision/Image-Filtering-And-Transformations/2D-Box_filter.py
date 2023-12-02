"""
2D BOX Filter

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
 ax[1].set_title('Box Filter (5,5)')
 ax[1].axis('off')
 plt.show()

def main():
 # read an image
 img = cv2.imread('../figures/flower.png')


 # To try different kernel, change size here.
 kernel_size = (5,5)

 # opencv has implementation for kernel based box blurring
 blur = cv2.blur(img,kernel_size)
 # Do plot
 plot_cv_img(img, blur)

 
if __name__ == '__main__':
 main()


