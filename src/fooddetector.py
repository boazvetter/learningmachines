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
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([50, 255, 50], np.uint8)

    # Create a mask for orange
    mask = cv2.inRange(img, lower, upper)
    return mask

img = cv2.imread('foodblock.png',1)
img = mask_img(img)
print(img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
