import cv2
import numpy as np

img = cv2.imread('~/Google Drive/Master/Learning Machines/learningmachines/robotviews/robotview1548272628-rgb.jpg')
img = cv2.resize(img,(240,320))
img=cv2.GaussianBlur(img, (9, 9), 0)
cv2.imwrite("testblur.jpg", img)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
minHSV = np.array([49, 70, 10])
maxHSV = np.array([90, 255, 255])
mask = cv2.inRange(hsv,minHSV,maxHSV)

cv2.imwrite('test1.jpg', mask)
