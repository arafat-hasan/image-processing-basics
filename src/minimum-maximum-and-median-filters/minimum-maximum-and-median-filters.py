#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: minimum-maximum-and-median-filters.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 12-12-19 16:29:53 (+06)
# LAST MODIFIED: 12-12-19 19:49:19 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 12-12-19     1.0         Deleted code is debugged code.
#
#               _/  _/_/_/_/  _/      _/  _/_/_/  _/      _/
#              _/  _/        _/_/    _/    _/    _/_/    _/
#             _/  _/_/_/    _/  _/  _/    _/    _/  _/  _/
#      _/    _/  _/        _/    _/_/    _/    _/    _/_/
#       _/_/    _/_/_/_/  _/      _/  _/_/_/  _/      _/
#
##############################################################################


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
import os
import sys


if __name__ == '__main__':
    path1 = '../../img/lennaSaltPepperNoisy.png'
    path2 = '../../img/bone-with-white-spot.jpg'
    path3 = '../../img/bone-with-black-spot.jpg'

    if os.path.isfile(path1):
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print ("[INFO] The file '" + path1 + "' does not exist.")
        sys.exit(0)

    if os.path.isfile(path2):
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print ("[INFO] The file '" + path2 + "' does not exist.")
        sys.exit(0)

    if os.path.isfile(path3):
        img3 = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print ("[INFO] The file '" + path3 + "' does not exist.")
        sys.exit(0)


    # Making each image of same size
    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))
    img3 = cv2.resize(img3, (512, 512))

    # Keeping a clone of each image without padding, to use later
    img1Cpy = img1.copy()
    img2Cpy = img2.copy()
    img3Cpy = img3.copy()

    (iH, iW) = img1.shape[:2]
    kW = 3 # Filter size
    pad = (kW - 1) // 2
    
    img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    img3 = cv2.copyMakeBorder(img3, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    outputMedian = np.zeros((iH, iW), dtype="float32")
    outputMin = np.zeros((iH, iW), dtype="float32")
    outputMax = np.zeros((iH, iW), dtype="float32")

    print("[INFO] Applying various filters...")
    print("[INFO] This may take a while...")
    for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                    roiMedian = img1[y - pad:y + pad + 1, x - pad:x + pad + 1]
                    roiMin = img2[y - pad:y + pad + 1, x - pad:x + pad + 1]
                    roiMax = img3[y - pad:y + pad + 1, x - pad:x + pad + 1]

                    outputMedian[y - pad, x - pad] = np.median(roiMedian)
                    outputMin[y - pad, x - pad] = np.amin(roiMin)
                    outputMax[y - pad, x - pad] = np.amax(roiMax)
    print("[INFO] Filter applying completed...")

    titles = ['Input Image', 'Input Image', 'Input Image', 'Median Filter', 'Minimum Filter', 'Maximum Filter']
    imgarr = [img1Cpy, img2Cpy, img3Cpy, outputMedian, outputMin, outputMax]
   
    for i in range(6):
        plt.subplot(2, 3,i+1)
        plt.imshow(imgarr[i], cmap = 'gray', vmin = 0, vmax = 255)
        plt.title(titles[i])
        plt.xticks ([])
        plt.yticks ([])

    plt.show ()

    print("[INFO] All operations completed successfully...")
    
