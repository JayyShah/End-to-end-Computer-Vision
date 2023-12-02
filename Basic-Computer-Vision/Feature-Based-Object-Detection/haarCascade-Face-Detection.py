"""

Detecting faces is a two-step process:

• First is to create a classifier with parameters for specific object detection. In our case,
it is face detection:
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

• In second step, for each image, it face detection is done using previously loaded
classifier parameters:
faces = face_cascade.detectMultiScale(gray)

"""


import numpy as np
import cv2
# create cascaded classifier with pre-learned weights
# For other objects, change the file here
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while(True):
 ret, frame = cap.read()
 if not ret:
 print("No frame captured")

 # frame = cv2.resize(frame, (640, 480))
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 # detect face
 faces = face_cascade.detectMultiScale(gray)
 # plot results
 for (x,y,w,h) in faces:
 cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
 cv2.imshow('img',frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
 break
cap.release()
cv2.destroyAllWindows()



"""

Here, we used a file "haarcascade_frontalface_default.xml" which contains classifier parameters
available at "https://github.com/opencv/opencv/tree/master/data/haarcascades". 

We have to download these cascade classifier files in order to run face detection. Also for detecting other
objects like eyes, smiles, and so on, we require similar files for use with OpenCV.

"""
