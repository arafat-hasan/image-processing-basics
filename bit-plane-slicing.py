#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: bit-plane-slicing.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 08-12-19 20:00:07 (+06)
# LAST MODIFIED: 23-12-19 00:53:37 (+06)
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
    path = 'img/coins.jpg'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    out = []
    for k in range(0, 8):
        # create an image for each k bit plane
        plane = np.full((img.shape[0], img.shape[1]), 2**k, np.uint8)
        # execute bitwise and operation
        res = cv2.bitwise_and(plane, img)
        # multiply ones (bit plane sliced) with 255 just for better visualization
        x = res * 255
        # append to the output list
        out.append(x)

    finalv = cv2.hconcat([out[3], out[2], out[1], out[0]])
    finalr = cv2.hconcat([out[7], out[6], out[5], out[4]])
    # Vertically concatenate
    final = cv2.vconcat([finalr, finalv])

    cv2.imshow('Bit Plane Slicing', final)
    cv2.waitKey(0)

    print("[INFO] All operations finished successfully...")
