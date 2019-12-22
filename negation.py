#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: negation.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 08-12-19 16:22:01 (+06)
# LAST MODIFIED: 23-12-19 00:46:26 (+06)
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
    path = 'img/pollen-image-plants.jpg'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    rows, cols = img.shape
    output = np.zeros((rows, cols), dtype='uint8')

    for row in range(rows):
        for col in range(cols):
            output[row, col] = 255 - img[row, col]

    titles = ['Input Image', 'Negative Image']
    imgarr = [img, output]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(imgarr[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations finished successfully...")
