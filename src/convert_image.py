import cv2
import numpy as np

img = cv2.imread('/Users/jellevanmil/Google Drive/Master/Learning Machines/learningmachines/robotviews/robotview1548271071-rgb.jpg')
# img = cv2.resize(img,(250,250))
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
minHSV = np.array([49, 70, 20])
maxHSV = np.array([94, 255, 255])
mask = cv2.inRange(hsv,minHSV,maxHSV)

cv2.imwrite('test.jpg', mask)
