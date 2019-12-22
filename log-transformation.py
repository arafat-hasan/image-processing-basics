#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: log-transformation.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 08-12-19 16:46:45 (+06)
# LAST MODIFIED: 08-12-19 16:52:22 (+06)
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
    path = 'img/misc/7.2.01.tiff'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    rows, cols = img.shape
    output = np.zeros((rows, cols), dtype='uint8')

    c = 25
    for row in range(rows):
        for col in range(cols):
            output[row, col] = c * np.log(img[row, col] + 1)

    titles = ['Input Image', 'Log Transformed Image']
    imgarr = [img, output]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(imgarr[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations finished successfully...")
