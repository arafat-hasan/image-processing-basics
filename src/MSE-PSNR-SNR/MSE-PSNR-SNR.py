#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: MSE-PSNR-SNR.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
# DATE CREATED: 06-11-19 14:06:21 (+06)
# LAST MODIFIED: 08-12-19 23:48:57 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 06-11-19     1.0         Deleted code is debugged code.
#
#               _/  _/_/_/_/  _/      _/  _/_/_/  _/      _/
#              _/  _/        _/_/    _/    _/    _/_/    _/
#             _/  _/_/_/    _/  _/  _/    _/    _/  _/  _/
#      _/    _/  _/        _/    _/_/    _/    _/    _/_/
#       _/_/    _/_/_/_/  _/      _/  _/_/_/  _/      _/
#
##############################################################################

import cv2
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import os
import sys


def fun(img):
    height, width = img.shape
    MSE_gauss = 0
    MSE_median = 0
    r = 0
    R = 0

    for i in range(0, height):
        for j in range(0, width):
            p = img[i, j]
            R = max(R, p)
            # r = r + p
            r = r + p * p
            p1 = randomNoisy[i, j]
            p2 = saltPepperNoisy[i, j]
            p3 = gausianBlur[i, j]
            p4 = medianBlur[i, j]

            MSE_gauss = MSE_gauss + (p1 - p3) * (p1 - p3)
            MSE_median = MSE_median + (p2 - p4) * (p2 - p4)

    MSE_gauss = MSE_gauss / (height * width)
    MSE_median = MSE_median / (height * width)

    PSNR_gauss = 10 * math.log10((R * R) / MSE_gauss)
    PSNR_median = 10 * math.log10((R * R) / MSE_median)

    r = r // (height * width)

    # SNR_gauss = 10 * math.log10( (r*r) / MSE_gauss )
    # SNR_median = 10 * math.log10( (r*r) / MSE_median )
    SNR_gauss = 10 * math.log10(r / MSE_gauss)
    SNR_median = 10 * math.log10(r / MSE_median)
    print('Gaussian MSE: ', MSE_gauss, '\tMedian MSE: ', MSE_median)
    print('Gaussian SNR: ', SNR_gauss, '\tMedian SNR: ', SNR_median)
    print('Gaussian PSNR: ', PSNR_gauss, '\tMedian PSNR: ', PSNR_median)


def saltPepperNoise(image, prob):
    # Thanks to ppk28
    # https://stackoverflow.com/a/27342545/7829174
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)  #uint8 - unsigned 8 bit integer

    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random(
            )  # generates a random number between (0.0 to 1.0)
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == '__main__':

    path = '../../img/lennaGray.png'

    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Image has been read successfully...")
    else:
        print("[INFO] The file '" + path + "' does not exist.")
        sys.exit(0)

    randomNoisy = random_noise(img, mode='gaussian', seed=None, clip=True)
    saltPepperNoisy = saltPepperNoise(img, 0.05)
    gausianBlur = cv2.GaussianBlur(randomNoisy, (5, 5), 0)
    medianBlur = cv2.medianBlur(saltPepperNoisy, 5)

    img = img.astype(float)
    randomNoisy = randomNoisy.astype(float)
    saltPepperNoisy = saltPepperNoisy.astype(float)
    gausianBlur = gausianBlur.astype(float)
    medianBlur = medianBlur.astype(float)

    fun(img)

    imgArr = [randomNoisy, saltPepperNoisy, gausianBlur, medianBlur]
    title = [
        'Gaussian Noisy', 'Salt Pepper Noisy', 'Gaussian Blur', 'Median Blur'
    ]

    for i in range(len(imgArr)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(imgArr[i], cmap='gray')
        plt.title(title[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("[INFO] All operations finished successfully...")
