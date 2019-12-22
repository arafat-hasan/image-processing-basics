#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: addition.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 06-12-19 17:49:10 (+06)
# LAST MODIFIED: 22-12-19 19:29:28 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 06-12-19     1.0         Deleted code is debugged code.
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
    path1 = 'img/rectangle-1.png'
    path2 = 'img/rectangle-2.png'

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
    output = np.zeros((rows, cols), dtype='uint8')

    for row in range(rows):
        for col in range(cols):
            tmp = int(img1[row, col]) + int(img2[row, col])
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
