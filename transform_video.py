import numpy as np
from imutils import perspective
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
 help="path to input video")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["input"])
# top left, bottom left, bottom right, top right
# https://github.com/jrosebr1/imutils/blob/master/demos/perspective_transform.py
pts = np.array([
    (279, 274),
    (327, 351),
    (640, 330),
    (627, 279)
])

while(True):
    ret, frame = vs.read()
    warped = perspective.four_point_transform(frame, pts)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('frame', frame)
    cv2.imshow('warped', warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
