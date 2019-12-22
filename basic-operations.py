#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: basic-operations.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 05-12-19 20:16:11 (+06)
# LAST MODIFIED: 22-12-19 17:26:01 (+06)
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
import os

if __name__ == '__main__':

    path = 'img/misc/house.tiff'

    if os.path.isfile(path):
        img = cv2.imread(path)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    cv2.imshow('Original', img)
    cv2.imwrite('NewHouseImage.png', img)

    arbitraryPixel = img[2, 3]
    print('An arbitrary Pixel: ', arbitraryPixel)

    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    print('Image Dimension: ', dimensions)
    print('Image Height: ', height)
    print('Image Width: ', width)
    print('Total Number of pixels :', img.size)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("[INFO] All operations finished successfully...")
