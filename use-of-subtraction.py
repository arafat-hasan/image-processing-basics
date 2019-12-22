#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: addition.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 06-12-19 17:49:10 (+06)
# LAST MODIFIED: 06-12-19 18:52:51 (+06)
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
        print("First image has been read sucessfully...")
    else:
        print ("The file '" + path1 + "' does not exist.")

    if os.path.isfile(path2):
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        print("Second image has been read sucessfully...")
    else:
        print ("The file '" + path2 + "' does not exist.")

    
    if img1.shape != img2.shape:
        print("Image sizes are not identical, addition is not possible.")
        print("Aborting program...")
        sys.exit(0)

    rows, cols = img1.shape
    output = np.zeros((rows, cols), dtype='uint8')

    for row in range(rows):
        for col in range(cols):
            tmp = int(img1[row, col]) - int(img2[row, col])
            output[row, col] = max(0, min(tmp, 255))

    titles = ['First Input Igmage', 'Second Input Image', 'Output Image']
    imgarr = [img1, img2, output]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(imgarr[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    print("All operations finished sucessfully...")

