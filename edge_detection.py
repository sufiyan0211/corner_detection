import cv2
import numpy as np
import matplotlib.pyplot as plt 
import time

def edge_detection_canny(img):
    new_img = img.copy()
    edges = cv2.Canny(img,127,127)
    # already edges is in gray format
    mask = cv2.bitwise_not(edges)
    new_img = cv2.bitwise_or(img,img,mask)
    return mask


cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)

    cv2.imshow('ei',frame)

    #time.sleep(2)
    frame = edge_detection_canny(frame)

    #time.sleep(2)
    cv2.imshow('video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


