#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: weber_ratio.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 05-12-19 21:42:24 (+06)
# LAST MODIFIED: 22-12-19 17:54:52 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 05-12-19     1.0         Deleted code is debugged code.
#
#               _/  _/_/_/_/  _/      _/  _/_/_/  _/      _/
#              _/  _/        _/_/    _/    _/    _/_/    _/
#             _/  _/_/_/    _/  _/  _/    _/    _/  _/  _/
#      _/    _/  _/        _/    _/_/    _/    _/    _/_/
#       _/_/    _/_/_/_/  _/      _/  _/_/_/  _/      _/
#
##############################################################################

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'img/weber-ratio.png'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    rows, cols = img.shape
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    weberConstant = 0.02
    weberMax = -999999.0
    print("[INFO] Operation on progress...")
    for row in range(rows):
        for col in range(cols):
            pix = img[row, col]
            for x, y in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
                         (-1, 0), (-1, 1)]:  # move all eight directions
                tmpX = row + x
                tmpY = col + y
                # Handle if out of array
                if (tmpX < 0 or tmpY < 0 or tmpX >= rows or tmpY >= cols):
                    continue

                tmpPix = img[tmpX, tmpY]
                brightnessDiff = abs(int(pix) - int(tmpPix))

                # weberRatioTmp is weber ratio of current working pixel
                if (pix == 0):
                    weberRatioTmp = (brightnessDiff) / (pix + 1)
                else:
                    weberRatioTmp = brightnessDiff / pix
                weberMax = max(weberMax, weberRatioTmp)

                # Set GREEN channel to maximum to detect edges if it cross weberConstant
                if (weberRatioTmp > weberConstant):
                    output[row, col, 1] = 255

    print("Maximum Weber's ratio: ", weberMax)

    titles = ['Original image', 'Original image with background edges']
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(titles[0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title(titles[1])
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print("[INFO] All operations finished successfully...")
