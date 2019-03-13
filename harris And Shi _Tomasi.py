import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner(img):
    new_img = img.copy()
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = np.float32(img_gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    new_img[dst>0.01*dst.max()] = [0,0,0]
    return new_img


def shi_tomasi_gftt(img):
    new_img = img.copy()
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_img,150,0.01,5)

    corners = np.int0(corners)
 
    for i in corners:
        x,y = i.ravel()
        cv2.circle(new_img,(x,y),3,(80,127,255),-1)

    return new_img





cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fout = cv2.VideoWriter('/home/sufiyan0211/Desktop/fg.mp4',fourcc,20.0,(width,height))


while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)

    #frame = harris_corner(frame)
    frame = shi_tomasi_gftt(frame)
    fout.write(frame)
    

    cv2.imshow('windows',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fout.release()
cap.release()
cv2.destroyAllWindows()