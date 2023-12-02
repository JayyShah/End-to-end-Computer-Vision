"""

Median-Filter

"""

# First input is added with channel wise random noise 

# read the image
flower = cv2.imread('../figures/flower.png')
# initialize noise image with zeros
noise = np.zeros(flower.shape[:2])
# fill the image with random numbers in given range
cv2.randu(noise, 0, 256)
# add noise to existing image, apply channel wise
noise_factor = 0.1
noisy_flower = np.zeros(flower.shape)
for i in range(flower.shape[2]):
 noisy_flower[:,:,i] = flower[:,:,i] + np.array(noise_factor*noise, dtype=np.int)
# convert data type for use
noisy_flower = np.asarray(noisy_flower, dtype=np.uint8)


# Created noisy image is used for median filtering as:

# apply median filter of kernel size 5
kernel_5 = 5
median_5 = cv2.medianBlur(noisy_flower,kernel_5)
# apply median filter of kernel size 3
kernel_3 = 3
median_3 = cv2.medianBlur(noisy_flower,kernel_3)