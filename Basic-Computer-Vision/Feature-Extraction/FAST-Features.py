"""

FAST - Features From Accelerated Segment Test

"""

"""

The steps to compute FAST features are as follows:
1. Initialize detector using cv2.FastFeatureDetector_create()
2. Setup threshold parameters for filtering detections
3. Setup flag if non-maximal suppression to be used for clearing neighbourhood regions
of repeated detections
4. Detect keypoints and plot them on the input image 

"""

def compute_fast_det(filename, is_nms=True, thresh = 10):
 """
 Reads image from filename and computes FAST keypoints.
 Returns image with keypoints
 filename: input filename
 is_nms: flag to use Non-maximal suppression
 thresh: Thresholding value
 """
 img = cv2.imread(filename)

 # Initiate FAST object with default values
 fast = cv2.FastFeatureDetector_create()
 # find and draw the keypoints
 if not is_nms:
 fast.setNonmaxSuppression(0)
 fast.setThreshold(thresh)
 kp = fast.detect(img,None)
 cv2.drawKeypoints(img, kp, img, color=(255,0,0))

 return img
