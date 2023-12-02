def compute_orb_keypoints(filename):
 """
 Takes in filename to read and computes ORB keypoints
 Returns image, keypoints and descriptors
 """
 img = cv2.imread(filename)

 # downsample image 4x
 img = cv2.pyrDown(img) # downsample 2x
 img = cv2.pyrDown(img) # downsample 4x
 # create orb object
 orb = cv2.ORB_create()

 # set parameters
 orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

 # detect keypoints
 kp = orb.detect(img,None)
 kp, des = orb.compute(img, kp)
 return img, kp, des
Using the previously computed keypoints and descriptors, the matching is done as:
def compute_img_matches(filename1, filename2, thres=10):
 """
 Extracts ORB features from given filenames
 Computes ORB matches and plot them side by side
 """
 img1, kp1, des1 = compute_orb_keypoints(filename1)
 img2, kp2, des2 = compute_orb_keypoints(filename2)

 matches = brute_force_matcher(des1, des2)
 draw_matches(img1, img2, kp1, kp2, matches, thres)

def brute_force_matcher(des1, des2):
 """
 Brute force matcher to match ORB feature descriptors
 """
 # create BFMatcher object
 bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
 # Match descriptors.
 matches = bf.match(des1,des2)
 # Sort them in the order of their distance.
 matches = sorted(matches, key = lambda x:x.distance)
 return matches
def draw_matches(img1, img2, kp1, kp2, matches, thres=10):
 """
 Utility function to draw lines connecting matches between two images.
 """
 draw_params = dict(matchColor = (0,255,0),
 singlePointColor = (255,0,0),
flags = 0)
 # Draw first thres matches.
 img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:thres],None, **draw_params)
 plot_img(img3)
def main():
 # read an image
 filename2 = '../figures/building_7.JPG'
 filename1 = '../figures/building_crop.jpg'
 compute_img_matches(filename1, filename2, thres=20)
if __name__ == '__main__':
 main()
