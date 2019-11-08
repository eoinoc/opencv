# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-d", "--darknet", type=str, default='../darknet',
	help="base path to Darknet directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
# https://pjreddie.com/darknet/
labelsPath = os.path.sep.join([args["darknet"], "data/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
 
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["darknet"], "yolov3.weights"])
configPath = os.path.sep.join([args["darknet"], "cfg/yolov3.cfg"])
 
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
