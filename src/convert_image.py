import cv2
import numpy as np

filename = '../robotviews/IMG_20190124_095125'
img = cv2.imread(filename + '.jpg')
img = cv2.resize(img,(250,250))
cv2.imwrite(filename + '-resize.jpg', img)
img=cv2.GaussianBlur(img, (9, 9), 0)
cv2.imwrite(filename + '-gaussian.jpg', img)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
minHSV = np.array([31, 42, 30])
maxHSV = np.array([96, 255, 255])
mask = cv2.inRange(hsv,minHSV,maxHSV)
print(np.sum(mask) / 255 )
cv2.imwrite(filename + '-mask.jpg', mask)
