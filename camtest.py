import numpy as np
import cv2

# From: https://www.hackster.io/mjrobot/automatic-vision-object-tracking-5575c4#toc-step-5--object-movement-tracking-6

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
