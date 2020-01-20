#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: division.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 08-12-19 13:37:58 (+06)
# LAST MODIFIED:
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 08-12-19     1.0         Deleted code is debugged code.
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
import math
import os
import sys

if __name__ == '__main__':
    path1 = '../../img/rectangle-div-1.png'
    path2 = '../../img/rectangle-div-2.png'

    if os.path.isfile(path1):
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        print("[INFO] First image has been read successfully...")
    else:
        print("[INFO] The file '" + path1 + "' does not exist.")
        sys.exit(0)

    if os.path.isfile(path2):
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Second image has been read successfully...")
    else:
        print("[INFO] The file '" + path2 + "' does not exist.")
        sys.exit(0)

    if img1.shape != img2.shape:
        print("Image sizes are not identical, resizing second image.")
        img2 = cv2.resize(img2, img1.shape[1], img1.shape[2])

    rows, cols = img1.shape
    output = np.zeros((rows, cols), dtype='float')
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    for row in range(rows):
        for col in range(cols):
            if img2[row, col] != 0:
                tmp = math.ceil(img1[row, col] / img2[row, col])
            else:  # division by zero wiil generate a huge number, here 255 is that huge
                tmp = 255
            output[row, col] = max(0, min(tmp, 255))

    titles = ['First Input Image', 'Second Input Image', 'Output Image']
    imgarr = [img1, img2, output]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(imgarr[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations finished successfully...")
