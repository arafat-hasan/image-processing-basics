#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: detect-edge-using-canny-sobel-prewitt.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 11-12-19 23:09:49 (+06)
# LAST MODIFIED: 12-12-19 00:59:08 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 11-12-19     1.0         Deleted code is debugged code.
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
import matplotlib.pyplot as plt
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
    path = '../../img/Valve_original_(1).png'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print ("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)


    imgGaussian = cv2.GaussianBlur(img, (3, 3), 0)
    print("[INFO] Gaussian operator applied...")
    imgCanny = cv2.Canny(img,100,200)
    print("[INFO] Canny operator applied...")

    horizontalSobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    verticalSobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # imgSobelX = cv2.Sobel(imgGaussian,cv2.CV_8U,1,0,ksize=3)
    # imgSobelY = cv2.Sobel(imgGaussian,cv2.CV_8U,0,1,ksize=3)
    imgSobelX = convolve(imgGaussian, horizontalSobel)
    imgSobelY = convolve(imgGaussian, verticalSobel)
    imgSobel = imgSobelX + imgSobelY
    print("[INFO] Sobel operator applied...")

    prewittKernelX = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    prewittKernelY = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # imgPrewittX = cv2.filter2D(imgGaussian, -1, prewittKernelX)
    # imgPrewittY = cv2.filter2D(imgGaussian, -1, prewittKernelY)
    imgPrewittX = convolve(imgGaussian, prewittKernelX)
    imgPrewittY = convolve(imgGaussian, prewittKernelY)
    imgPrewitt = imgPrewittX + imgPrewittY
    print("[INFO] Prewitt operator applied...")
    
    titles = ['Input Image', 'Canny Edge', 'Sobel Edge','Prewitt Edge']
    imgarr = [img, imgCanny, imgSobel, imgPrewitt]
   
    plt.figure(figsize=(20,20))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(imgarr[i], cmap='gray', vmin =0, vmax =255)
        plt.title(titles[i])
        plt.xticks ([])
        plt.yticks ([])

    plt.show ()

    print("[INFO] All operations completed successfully...")
