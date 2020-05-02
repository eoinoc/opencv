# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="videos/red-car-turning.mp4",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

confidenceThreshold = 0.5
modelConfiguration = '../darknet/cfg/yolov3-tiny.cfg'
modelWeights = '../darknet/yolov3-tiny.weights'
labelsPath = '../darknet/data/coco.names'
labels = open(labelsPath).read().strip().split('\n')

# OpenCV Net class, to manipulate neural nets
# https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# determine only the *output* layer names that we need from YOLO
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, use the sample
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])
(W, H) = (None, None)

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
 
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    # 288 / 416
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (288, 288), 
        swapRB = True, crop = False)
    net.setInput(blob)

    # Runs forward pass to compute output of layer
    outputsOfLayer = net.forward(layerNames)

    boxes = []
    confidences = []
    classIDs = []

    for output in outputsOfLayer:
        for detection in output:
            # returns the probability score for each classes
            # 5: means range of "5 to end"
            scores = detection[5:]

            classID = np.argmax(scores)
            # TODO it's always returning index 0 now, so doesn't 
            # seem to be classifying correctly
            # if classID not in ["car"]:
            #     print(classID)
            #     continue
            
            confidence = scores[classID]
            if confidence > confidenceThreshold:
       
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
        
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                radius = 50
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
