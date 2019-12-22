#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: intensity-level-slicing.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 08-12-19 17:18:41 (+06)
# LAST MODIFIED: 23-12-19 00:52:40 (+06)
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
import os
import sys

if __name__ == '__main__':
    path = 'img/cameraman.tif'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    rows, cols = img.shape
    outputWithBack = np.zeros((rows, cols), dtype='uint8')
    outputWithoutBack = np.zeros((rows, cols), dtype='uint8')

    # Specify the min and max range
    min_range = 5
    max_range = 55
    for row in range(rows):
        for col in range(cols):
            if img[row, col] > min_range and img[row, col] < max_range:
                outputWithBack[row, col] = 255
                outputWithoutBack[row, col] = 255
            else:
                outputWithBack[row, col] = img[row, col]
                outputWithoutBack[row, col] = 0

    titles = [
        'Input Image', 'Intensity-level Slicing with Background',
        'Intensity-level Slicing without Background'
    ]
    imgarr = [img, outputWithBack, outputWithoutBack]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(imgarr[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations finished successfully...")
