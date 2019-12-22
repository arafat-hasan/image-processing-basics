#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: line-detection.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 20-11-19 10:35:44 (+06)
# LAST MODIFIED: 12-12-19 21:58:09 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 14-11-19     1.0         Deleted code is debugged code.
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
from skimage.exposure import rescale_intensity
import os
import sys

def convolve(image, kernel):
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			k = (roi * kernel).sum()
			output[y - pad, x - pad] = k

	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	return output


if __name__ == '__main__':

    path = 'img/Testbuilding.png'

    if os.path.isfile(path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print ("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)


    horizontalMask = np.array((
            [-1, -1, -1],
            [2, 2, 2],
            [-1, -1, -1]), dtype="int")

    verticalMask = np.array((
            [-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1]), dtype="int")

    oblique45P = np.array((
            [-1, -1, 2],
            [-1, 2, -1],
            [2, -1, -1]), dtype="int")

    oblique45N = np.array((
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2]), dtype="int")


    # construct the kernel bank, a list of kernels we're going
    # to apply using custom `convole` function
    kernelBank = (
            ("Horizontal Mask", horizontalMask),
            ("Vertical Mask", verticalMask),
            ("Oblique +45 Degree", oblique45P),
            ("Oblique -45 Degree", oblique45N)
    )

    # loop over the kernels
    convoleOutput = []
    titles = []
    for (kernelName, kernel) in kernelBank:
        print("[INFO] Applying {} kernel...".format(kernelName))
        convoleOutput.append(convolve(image, kernel))
        titles.append(kernelName)


    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray', vmin = 0, vmax = 255)
    plt.title("Original Image")
    plt.xticks([]),plt.yticks([])

    for i in np.arange(4, 8):
        plt.subplot(2, 4, i+1)
        plt.imshow(convoleOutput[i-4], cmap='gray', vmin = 0, vmax = 255)
        plt.title(titles[i-4])
        plt.xticks([]),plt.yticks([])
    
    plt.show()

    print("[INFO] All operations completed successfully...")
