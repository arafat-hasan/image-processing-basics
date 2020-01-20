#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: ringing-artifacts.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 20-11-19 14:43:54 (+06)
# LAST MODIFIED: 22-12-19 01:19:54 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 20-11-19     1.0         Deleted code is debugged code.
#
#               _/  _/_/_/_/  _/      _/  _/_/_/  _/      _/
#              _/  _/        _/_/    _/    _/    _/_/    _/
#             _/  _/_/_/    _/  _/  _/    _/    _/  _/  _/
#      _/    _/  _/        _/    _/_/    _/    _/    _/_/
#       _/_/    _/_/_/_/  _/      _/  _/_/_/  _/      _/
#
##############################################################################

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':

    path = '../../img/aerials/2.2.02.tiff'
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitudeSpectrum = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)

    outputImgArr = [img, magnitudeSpectrum]
    maskSize = [5, 15, 25, 35, 50, 80]
    titles = ['Input Image', "Magnitude Spectrum"
             ] + ['Mask Size: ' + str(i) for i in maskSize]

    for n in maskSize:
        mask[crow - n:crow + n, ccol - n:ccol + n] = 1
        # apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        imgTmp = cv2.idft(f_ishift)
        outputImgArr.append(cv2.magnitude(imgTmp[:, :, 0], imgTmp[:, :, 1]))

    for i in range(len(outputImgArr)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(outputImgArr[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations completed successfully...")
