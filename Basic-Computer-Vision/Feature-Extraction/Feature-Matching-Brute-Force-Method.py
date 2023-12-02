"""

Using ORB features, we can do feature matching in a brute force way as follows:
• Compute features in each image (template and target).
• For each feature in a template, compare all the features in the target detected
previously. The criterion is set using a matching score.
• If the feature pair passes the criterion, then they are considered a match.
• Draw matches to visualize.

"""

def compute_orb_keypoints(filename):
 """
 Takes in filename to read and computes ORB keypoints
 Returns image, keypoints and descriptors
 """
 img = cv2.imread(filename)
 # create orb object
 orb = cv2.ORB_create()

 # set parameters
 orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

 # detect keypoints
 kp = orb.detect(img,None)
 # using keypoints, compute descriptor
 kp, des = orb.compute(img, kp)
 return img, kp, des


"""

Once we have keypoints and descriptors from each of the images, we can use them to
compare and match.
Matching keypoints between two images is a two-step process:

• Create desired kind of matcher specifying the distance metric to be used. Here we
will use Brute-Force Matching with Hamming distance:
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

• Using descriptors for keypoints from each image, perform matching as:
matches = bf.match(des1,des2)

"""
def brute_force_matcher(des1, des2):
 """
 Brute force matcher to match ORB feature descriptors
 des1, des2: descriptors computed using ORB method for 2 images
 returns matches
 """
 # create BFMatcher object
 bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
 # Match descriptors.
 matches = bf.match(des1,des2)
 # Sort them in the order of their distance.
 matches = sorted(matches, key = lambda x:x.distance)
 return matches


 def compute_img_matches(filename1, filename2, thres=10):
 """
 Extracts ORB features from given filenames
 Computes ORB matches and plot them side by side
 """
 img1, kp1, des1 = compute_orb_keypoints(filename1)
 img2, kp2, des2 = compute_orb_keypoints(filename2)

 matches = brute_force_matcher(des1, des2)
 draw_matches(img1, img2, kp1, kp2, matches, thres)

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
 filename1 = '../figures/building_crop.jpg'
 filename2 = '../figures/building.jpg'
 compute_img_matches(filename1, filename2)

if __name__ == '__main__':
 main()
