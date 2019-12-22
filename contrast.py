#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: contrast.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 05-12-19 17:23:22 (+06)
# LAST MODIFIED: 22-12-19 17:13:01 (+06)
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
import numpy as np
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':
    path = 'img/misc/7.1.01.tiff'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    rows, cols = img.shape
    output = np.zeros((rows, cols), dtype='float')
    img = img.astype(float)  # To get rid of from overflow
    contrastConstant = 1.7

    for row in range(rows):
        for col in range(cols):
            tmp = img[row, col] * contrastConstant
            output[row, col] = max(0, min(tmp, 255))

    # This is more pythonic way for changing  brightness
    # output = img * contrastConstant  # No need to iterate over every pixel
    # lowerbound, upperbound = 0, 255
    # np.clip(output, lowerbound, upperbound, out=output)  # Numpy do the bounding

    print('Original Contrast: ', np.amax(img) - np.amin(img))
    print('Increased Contrast: ', np.amax(output) - np.amin(output))

    titles = ['Original Contrast', 'Increased Contrast']
    imgarr = [img, output]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(imgarr[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations finished successfully...")
