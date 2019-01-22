#!/usr/bin/env python2

from __future__ import print_function
import cv2
import glob
import numpy as np
import os

# def read_imgs(img):
#     '''
#     Read images from the given folder and return them in a list.
#     https://www.quora.com/How-can-I-read-multiple-images-in-Python-presented-in-a-folder
#     '''
#     #img_path = os.path.join(img_dir, '*g')
#     files = glob.glob(os.getcwd())
#     print("file: ", file)
#     imgs = []
#     for f in files:
#         print(f)
#         img = cv2.imread(f)
#         imgs.append(img)
#     return imgs

# someimg = read_imgs()
# print(someimg)


def mask_img(img):
    # Lower and upper boundary of green
    lower = np.array([0, 0, 1], np.uint8)
    upper = np.array([50, 255, 50], np.uint8)

    # Create a mask for orange
    mask = cv2.inRange(img, lower, upper)
    return mask

img = cv2.imread('foodblock.png',1)

STATE_LABEL_FOOD = ["Food Top Left", "Food Top Center", "Food Top Right", "Food Middle Left", "Food Middle Center", "Food Middle Right", "Food Bottom Left", "Food Middle Center", "Food Middle Right"]

def get_state_food():
#Subimages:
#[0 1 2
# 3 4 5
# 6 7 8]
	img = cv2.imread('foodblock.png',1)
	greencount = []
	for i in range(3):
		for j in range(3):
			part = len(img)/3
			sub_image = img[int(part*i):int(part*(i+1)), int(part*j):int(part*(j+1))]
			sub_image = mask_img(sub_image)
			greencount.append(np.count_nonzero(sub_image))
			s = '%s %s' % (i, j)			
			cv2.imshow(s, sub_image)			
	return greencount.index(max(greencount))

print(get_state_food())

cv2.imshow('image', img)
masked_full = mask_img(img)
cv2.imshow('masked', masked_full)
cv2.waitKey(0)
cv2.destroyAllWindows()
