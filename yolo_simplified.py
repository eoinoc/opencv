# https://github.com/mohitwildbeast/YOLO-v3-Object-Detection/blob/master/yolo_detection_video.py
# Basically looks to be a copy of:
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import time

confidenceThreshold = 0.5
NMSThreshold = 0.3

modelConfiguration = '../darknet/cfg/yolov3-tiny.cfg'
modelWeights = '../darknet/yolov3-tiny.weights'
#modelConfiguration = '../darknet/cfg/yolov3.cfg'
#modelWeights = '../darknet/yolov3.weights'

labelsPath = '../darknet/data/coco.names'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

outputLayer = net.getLayerNames()
outputLayer = [outputLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = VideoStream(usePiCamera=True).start()
# vs = VideoStream(usePiCamera=True, resolution=(320, 240)).start()
# Wait for camera to become available
time.sleep(2.0)

writer = None
(W, H) = (None, None)

fps = FPS().start()
count = 0
while True:
    frame = vs.read()
    frame = frame
    
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    # 416
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (288, 288), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(outputLayer)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    #Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    if(len(detectionNMS) > 0):
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

