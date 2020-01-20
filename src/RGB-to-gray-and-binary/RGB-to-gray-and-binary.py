#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: RGB-to-gray-and-binary.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 06-12-19 02:07:57 (+06)
# LAST MODIFIED: 22-12-19 19:16:31 (+06)
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
import math
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    path = '../../img/misc/4.1.08.tiff'

    if os.path.isfile(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    rows, cols, ch = img.shape
    img = img.astype('float')
    imgGraygen = np.zeros((rows, cols), dtype='uint8')
    imgGrayweighted = np.zeros((rows, cols), dtype='uint8')
    imgBinary = np.zeros((rows, cols), dtype='bool_')
    thresh = 170

    for row in range(rows):
        for col in range(cols):
            pixel = img[row, col]
            avggen = int(math.ceil(pixel[0] + pixel[1] + pixel[2]) / 3)
            avgweighted = int(
                math.ceil(pixel[0] * 0.299 + pixel[1] * 0.587 +
                          pixel[2] * 0.144))
            imgGraygen[row, col] = max(0, min(avggen, 255))
            imgGrayweighted[row, col] = max(0, min(avgweighted, 255))
            imgBinary[row, col] = False if avgweighted > thresh else True

    # ======================================================
    # IMPLEMENTATION USING OPENCV LIBRARY
    # ======================================================
    # (thresh, imgBinary) = cv2.threshold(imgGrayweighted, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    titles = [
        'Original Image', 'Average Grayscale Image', 'Weighted Grayscale Image',
        'Binary Image'
    ]
    imgarr = [img, imgGraygen, imgGrayweighted, imgBinary]

    plt.subplot(1, 4, 1)
    plt.imshow(np.uint8(imgarr[0]))
    plt.title(titles[0])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 4, 2)
    plt.imshow(imgarr[1], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[1])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 4, 3)
    plt.imshow(imgarr[2], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[2])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 4, 4)
    plt.imshow(imgarr[3], cmap='binary')
    plt.title(titles[3])
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print("[INFO] All operations finished successfully...")
