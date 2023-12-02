"""
Median - Filter - Varying - Kernel - Size

"""

def plot_cv_img(input_image, output_image1, output_image2, output_image3):
 """
 Converts an image from BGR to RGB and plots
 """
 fig, ax = plt.subplots(nrows=1, ncols=4)
 ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
 ax[0].set_title('Input Image')
 ax[0].axis('off')

 ax[1].imshow(cv2.cvtColor(output_image1, cv2.COLOR_BGR2RGB))
 ax[1].set_title('Median Filter (3,3)')
 ax[1].axis('off')
 ax[2].imshow(cv2.cvtColor(output_image2, cv2.COLOR_BGR2RGB))
 ax[2].set_title('Median Filter (5,5)')
 ax[2].axis('off')
 ax[3].imshow(cv2.cvtColor(output_image3, cv2.COLOR_BGR2RGB))
 ax[3].set_title('Median Filter (7,7)')
 ax[3].axis('off')

 plt.show()

def main():
 # read an image
 img = cv2.imread('../figures/flower.png')
 # compute median filtered image varying kernel size
 median1 = cv2.medianBlur(img,3)
 median2 = cv2.medianBlur(img,5)
 median3 = cv2.medianBlur(img,7)


 # Do plot
 plot_cv_img(img, median1, median2, median3)
if __name__ == '__main__':
 main()
