"""
The method is divided into two stages:

1. In the first stage, the features are extracted from an image and Region of
Interests (ROI) are proposed. ROIs consists of a possible box where an object might
be in the image.

2. The second stage uses features and ROIs to compute final bounding boxes and class
probabilities for each of the boxes. These together constitute the final output.

------------------------------------------------------------------------------------------------------------------------------------------------

# An input image is used to extract features and a region proposals. These extracted features and proposals are used
# together to compute predicted bounding boxes and class probabilities for each box.

# For Additional Info, Check the image in the images folder.

-------------------------------------------------------------------------------------------------------------------------------------------------

Overall method is considered two-stage because during training the model will first learn to produce ROIs using a sub-model called Region
Proposal Network (RPN). It will then learn to produce correct class probabilities and bounding box locations using ROIs and features.

RPN layer uses feature layer as input creates a proposal for bounding boxes and corresponding probabilities.


--------------------------------------------------------------------------------------------------------------------------------------------------

Cloning this repository, as it will contain most of the required codes:

git clone https://github.com/tensorflow/models.git
cd models/research


Download the Pre-Trained model from 'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_24_10_2017.tar.gz'
This completes the downloading of the Pre-Trained model.

---------------------------------------------------------------------------------------------------------------------------------------------------

These two steps have to be performed each time we launch a Terminal shell:
• At first, we will compile protobuf files, as TensorFlow uses them to serialize structured
data:
protoc object_detection/protos/*.proto --python_out=.

• Also, run in the research folder:
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

"""


# Let's begin with loading libs that will be used here:
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


# In order to load a pre-trained model for prediction:
# load graph
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


#  It can be used to load the model Faster R-CNN with the ResNet-101 feature extractor
# pre-trained on MSCOCO dataset:
load_and_create_graph('faster_rcnn_resnet101_coco_2017_11_08/frozen_inference_graph.pb')

# Now, let's set up labels to display in our figure using MSCOCO labels:
# load labels for classes output

path_to_labels = os.path.join('data', 'mscoco_label_map.pbtxt')
# pre-training was done on 90 categories
nb_classes = 90
label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map,
 max_num_classes=nb_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Before final predictions, we will set up the utility function as:

def read_cv_image(filename):
 """
 Reads an input color image and converts to RGB order
 Returns image as an array
 """
 img = cv2.imread(filename)
 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 return img

# Following is utility function to display bounding boxes using matplotib:

def show_mpl_img_with_detections(img, dets, scores,
 classes, category_index,
 thres=0.6):
 """
 Applies thresholding to each box score and
 plot bbox results on image.
 img: input image as numpy array
 dets: list of K detection outputs for given image.(size:[1,K])
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

# Using this setup, we can do predictions on the input image.In the Following snippet, doing predictions on the input image as well as displaying the results.

# A Tensorflow session and run the graph in sess.run to compute bounding boxes, scores for each box, the class prediction for boxes and number of detections:

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
 image = read_cv_image(os.path.join(image_dir, 'cars2.png'))
 input_img = image[np.newaxis, :, :, :]

 # To compute prediction time
 start = time.time()
 # Run prediction and get 4 outputs
 (boxes, scores, classes, num) = sess.run(
 [detection_boxes, detection_scores, detection_classes, num_detections],
 feed_dict={image_tensor: input_img})
 end = time.time()
 print("Prediction time:",end-start,"secs for ", num[0], "detections")
 # display results
 show_mpl_img_with_detections(image, boxes[0],scores[0], classes[0],category_index,
thres=0.6)

