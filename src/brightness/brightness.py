#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: brightness.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 02-12-19 23:18:44 (+06)
# LAST MODIFIED: 22-12-19 16:23:28 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 02-12-19     1.0         Deleted code is debugged code.
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
    path = '../../img/lennaGray.png'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    rows, cols = img.shape
    output = np.zeros((rows, cols), dtype='float')
    img = img.astype(float)  # To get rid of from overflow
    brightnessConstant = 70

    for row in range(rows):
        for col in range(cols):
            tmp = img[row, col] - brightnessConstant
            output[row, col] = max(0, min(tmp, 255))

    # This is more pythonic way for changing  brightness
    # output = img - brightnessConstant  # No need to iterate over every pixel
    # lowerbound, upperbound = 0, 255
    # np.clip(output, lowerbound, upperbound, out=output)  # Numpy do the bounding

    print('Original Brightness: ', np.mean(img))
    print('Changed Brightness: ', np.mean(output))

    titles = ['Original Brightness', 'Changed Brightness']
    imgarr = [img, output]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(imgarr[i], cmap='gray', vmin = 0, vmax = 255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations finished successfully...")
