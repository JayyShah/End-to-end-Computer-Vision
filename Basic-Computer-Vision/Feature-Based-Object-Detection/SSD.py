"""

The prediction time is reduced byremoving the intermediate stage and the training is always end-to-end.
These networks have shown effectiveness by running on smartphones as well as low-end computation units:

------------------------------------------------------------------------------------------------------------------------------

To further increase the speed for detection, the model also uses a technique called nonmaximal suppression.
This will suppress all the Bounding Box which do not have a maximum score in a given region and for a given category.
 As a result, the total output boxes from the MultiBox Layer are reduced significantly and thus we have only high scored
detections per class in an image.

-------------------------------------------------------------------------------------------------------------------------------

Download the Pre-Trained from "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
tar -xvf ssd_inception_v2_coco_2017_11_17.tar.gz"

-------------------------------------------------------------------------------------------------------------------------------

• First, we will compile the protobuf files:
protoc object_detection/protos/*.proto --python_out=.

• Also, run in the research folder:
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

"""


# Let's begin with loading libraries:

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
# inside jupyter uncomment next line
# %matplotlib inline
import random
import time
from utils import label_map_util


# The following code reads pre-trained model. In TensorFlow, these models are usually
# saved as protobuf in .pbformat. Also, note that if there are other formats of pretrained model files, then we may have to read accordingly

def load_and_create_graph(path_to_pb):
 """
 Loads pre-trained graph from .pb file.
 path_to_pb: path to saved .pb file
 Tensorflow keeps graph global so nothing is returned
 """
 with tf.gfile.FastGFile(path_to_pb, 'rb') as f:
 # initialize graph definition
 graph_def = tf.GraphDef()
 # reads file
 graph_def.ParseFromString(f.read())
 # imports as tf.graph
 _ = tf.import_graph_def(graph_def, name='')


# For using our input image, the following block reads an image from a given path to a file:

def read_cv_image(filename):
 """
 Reads an input color image and converts to RGB order
 Returns image as an array
 """
 img = cv2.imread(filename)
 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 return img

# The last utility function is for the output display of the bounding box around the
# predicted object with the class name and detection score for each box:

def show_mpl_img_with_detections(img, dets, scores,
 classes, category_index,
 thres=0.6):
 """
 Applies thresholding to each box score and
 plot bbox results on image.
 img: input image as numpy array
 dets: list of K detection outputs for given image. (size:[1,K] )
 scores: list of detection score for each detection output(size: [1,K]).
 classes: list of predicted class index(size: [1,K])
 category_index: dictionary containing mapping from class index to class name.
 thres: threshold to filter detection boxes:(default: 0.6)
 By default K:100 detections
 """
 # plotting utilities from matplotlib
 plt.figure(figsize=(12,8))
 plt.imshow(img)
 height = img.shape[0]
 width = img.shape[1]
 # To use common color of one class and different for different classes
 colors = dict()
 # iterate over all proposed bbox
 # choose whichever is more than a threshold
 for i in range(dets.shape[0]):
 cls_id = int(classes[i])
 # in case of any wrong prediction for class index
 if cls_id >= 0:

 score = scores[i]
 # score for a detection is more than a threshold
 if score > thres:
 if cls_id not in colors:
 colors[cls_id] = (random.random(),
 random.random(),
 random.random())
 xmin = int(dets[i, 1] * width)
 ymin = int(dets[i, 0] * height)
xmax = int(dets[i, 3] * width)
 ymax = int(dets[i, 2] * height)
 rect = plt.Rectangle((xmin, ymin), xmax - xmin,
 ymax - ymin, fill=False,
 edgecolor=colors[cls_id],
 linewidth=2.5)
 plt.gca().add_patch(rect)
 # to plot class name and score around each detection box
class_name = str(category_index[cls_id]['name'])

 plt.gca().text(xmin, ymin - 2,
 '{:s} {:.3f}'.format(class_name, score),
 bbox=dict(facecolor=colors[cls_id], alpha=0.5),
fontsize=8, color='white')
 plt.axis('off')
 plt.show()
 return

# load pre-trained model
load_and_create_graph('ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb')

# Before using the model to start to make predictions on the input image, Create a dictionary map of class index
# to pre-defined class names. 

# load labels for classes output
path_to_labels = os.path.join('data', 'mscoco_label_map.pbtxt')
nb_classes = 90
label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map,
 max_num_classes=nb_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


image_dir = 'test_images/'
# create graph object from previously loaded graph
# tensorflow previously loaded graph as default
graph=tf.get_default_graph()
# launch a session to run this graph
with tf.Session(graph=graph) as sess:
 # get input node
 image_tensor = graph.get_tensor_by_name('image_tensor:0')

 # get output nodes
 detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
 detection_scores = graph.get_tensor_by_name('detection_scores:0')
 detection_classes = graph.get_tensor_by_name('detection_classes:0')
 num_detections = graph.get_tensor_by_name('num_detections:0')

 # read image from file and pre-process it for input.
 # Note: we can do this outside session scope too.
 image = read_cv_image(os.path.join(image_dir, 'person1.png'))
 # Input Shape : [N, Width,Height,Channels],
 # where N=1, batch size
 input_img = image[np.newaxis, :, :, :]

 # To compute prediction time
 start = time.time()
 # Run prediction and get 4 outputs
 (boxes, scores, classes, num) = sess.run(
 [detection_boxes, detection_scores, detection_classes, num_detections],
 feed_dict={image_tensor: input_img})
 end = time.time()
 print("Prediction time:",end-start,"secs for ", num, "detections")

 # display results with score threshold of 0.6
 # Since only one image is used , hence we use 0 index for outputs
 show_mpl_img_with_detections(image, boxes[0],scores[0], classes[0], thres=0.6)

 # To display the results, iterate simultaneously on images

 for i in range(nb_inputs):
 show_mpl_img_with_detections(images[i], boxes[i],scores[i], classes[i], thres=0.6)


 """

 TO show the comparision with two-Stage detector, for the same input.
 One-Stage Detector such as SSD is good for large objects but fails to recognixze small objects such as people.