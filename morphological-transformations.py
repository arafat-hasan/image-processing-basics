#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: line-detection.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 20-11-19 10:35:44 (+06)
# LAST MODIFIED: 12-12-19 21:58:09 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 14-11-19     1.0         Deleted code is debugged code.
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
from matplotlib import pyplot as plt
import os
import sys


if __name__ == '__main__':

    path = 'img/finger-print.jpg'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print ("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)


    ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


    titles = ['Main Image', 'Erosion on Main Image', 'Dilation on Erosion', 'Opening on Main Image', 'Closing on Opening']
    imgArr = [thresh, erosion, dilation, opening, closing]

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(imgArr[i], cmap='gray', vmin = 0, vmax = 255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    
    plt.show()

    print("[INFO] All operations completed successfully...")
