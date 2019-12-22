#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: alpha-blending.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 07-12-19 12:39:50 (+06)
# LAST MODIFIED: 23-12-19 00:25:11 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 07-12-19     1.0         Deleted code is debugged code.
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
    path1 = 'img/chair-1.png'
    path2 = 'img/dog-main-1.png'
    path3 = 'img/dog-alpha-1.png'

    if os.path.isfile(path1):
        background = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Background image has been read sucessfully...")
    else:
        print("[INFO] The file '" + path1 + "' does not exist.")
        sys.exit(0)

    if os.path.isfile(path2):
        foreground = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Foreground image has been read sucessfully...")
    else:
        print("[INFO] The file '" + path2 + "' does not exist.")
        sys.exit(0)

    if os.path.isfile(path3):
        alphaMask = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Alpha Mask image has been read sucessfully...")
    else:
        print("[INFO] The file '" + path3 + "' does not exist.")
        sys.exit(0)

    if foreground.shape != background.shape or background.shape != alphaMask.shape:
        print(
            "[INFO] Image sizes are not identical, resizing possible but not recommended."
        )
        print("[INFO] Aborting program...")
        sys.exit(0)

    alphaCpy = alphaMask
    rows, cols = background.shape
    output = np.zeros((rows, cols), dtype='float')

    foreground = foreground.astype(float)
    background = background.astype(float)
    alphaMask = alphaMask.astype(float) / 255

    print("[INFO] Blending on progress...")
    for row in range(rows):
        for col in range(cols):
            tmp = alphaMask[row, col] * foreground[row, col] + \
                    (1 - alphaMask[row, col]) * background[row, col]
            output[row, col] = max(0, min(tmp, 255))
    print("[INFO] Done...")

    titles = [
        'Background Image', 'Foreground Image', 'Alpha Mask', 'Output Image'
    ]
    imgarr = [background, foreground, alphaCpy, output]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(imgarr[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations finished successfully...")
