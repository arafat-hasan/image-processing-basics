#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FILE: anisotropic-diffusion.py
#
# @author: Arafat Hasan Jenin <opendoor.arafat[at]gmail[dot]com>
#
#           Original MATLAB code by Peter Kovesi
#           School of Computer Science & Software Engineering
#           The University of Western Australia
#           pk @ csse uwa edu au
#           <http://www.csse.uwa.edu.au>
#
#           Translated to Python and optimised by Alistair Muldal
#           Department of Pharmacology
#           University of Oxford
#           <alistair.muldal@pharm.ox.ac.uk>
#
#           More dynamic python code by Arafat Hasan
#           Department of Computer Science and Engineering
#           Mawlana Bhashani Science and Technology
#           Tanggail-1902, Bangladesh
#           <opendoor.arafata[at]gmail[dot]com>
#
# DATE CREATED: 21-01-20 21:55:31 (+06)
# LAST MODIFIED: 24-01-20 19:22:12 (+06)
#
# DEVELOPMENT HISTORY:
# Date         Version     Description
# --------------------------------------------------------------------
# 21-01-20     1.0         Dynamic and more usable code
#
#               _/  _/_/_/_/  _/      _/  _/_/_/  _/      _/
#              _/  _/        _/_/    _/    _/    _/_/    _/
#             _/  _/_/_/    _/  _/  _/    _/    _/  _/  _/
#      _/    _/  _/        _/    _/_/    _/    _/    _/_/
#       _/_/    _/_/_/_/  _/      _/  _/_/_/  _/      _/
#
##############################################################################

import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.filters as flt
import matplotlib.animation as animation
from collections import Counter
import warnings


def anisodiff3(stack,
               niter=1,
               kappa=50,
               gamma=0.1,
               step=(1., 1., 1.),
               option=1,
               ploton=False):
    """
	3D Anisotropic diffusion.

	Usage:
	stackout = anisodiff(stack, niter, kappa, gamma, option)

	Arguments:
	        stack  - input stack
	        niter  - number of iterations
	        kappa  - conduction coefficient 20-100 ?
	        gamma  - max value of .25 for stability
	        step   - tuple, the distance between adjacent pixels in (z,y,x)
	        option - 1 Perona Malik diffusion equation No 1
	                 2 Perona Malik diffusion equation No 2
	        ploton - if True, the middle z-plane will be plotted on every 
	        	 iteration

	Returns:
	        stackout   - diffused stack.

	kappa controls conduction as a function of gradient.  If kappa is low
	small intensity gradients are able to block conduction and hence diffusion
	across step edges.  A large value reduces the influence of intensity
	gradients on conduction.

	gamma controls speed of diffusion (you usually want it at a maximum of
	0.25)

	step is used to scale the gradients in case the spacing between adjacent
	pixels differs in the x,y and/or z axes

	Diffusion equation 1 favours high contrast edges over low contrast ones.
	Diffusion equation 2 favours wide regions over smaller ones.

	Reference: 
	P. Perona and J. Malik. 
	Scale-space and edge detection using ansotropic diffusion.
	IEEE Transactions on Pattern Analysis and Machine Intelligence, 
	12(7):629-639, July 1990.

	Original MATLAB code by Peter Kovesi  
	School of Computer Science & Software Engineering
	The University of Western Australia
	pk @ csse uwa edu au
	<http://www.csse.uwa.edu.au>

	Translated to Python and optimised by Alistair Muldal
	Department of Pharmacology
	University of Oxford
	<alistair.muldal@pharm.ox.ac.uk>

	June 2000  original version.       
	March 2002 corrected diffusion eqn No 2.
	July 2012 translated to Python
	"""

    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        showplane = stack.shape[0] // 2

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(stack[showplane, ...].squeeze(), interpolation='nearest')
        ih = ax2.imshow(stackout[showplane, ...].squeeze(),
                        interpolation='nearest',
                        animated=True)
        ax1.set_title("Original stack (Z = %i)" % showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in np.arange(1, niter):

        # calculate the diffs
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD / kappa)**2.) / step[0]
            gS = np.exp(-(deltaS / kappa)**2.) / step[1]
            gE = np.exp(-(deltaE / kappa)**2.) / step[2]
        elif option == 2:
            gD = 1. / (1. + (deltaD / kappa)**2.) / step[0]
            gS = 1. / (1. + (deltaS / kappa)**2.) / step[1]
            gE = 1. / (1. + (deltaE / kappa)**2.) / step[2]

        # update matrices
        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma * (UD + NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(stackout[showplane, ...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return stackout


def anisodiff(img,
              niterlist=[1],
              kappa=50,
              gamma=0.1,
              step=(1., 1.),
              sigma=0,
              option=1):
    """
	Anisotropic diffusion.

	Usage:
	imgout = anisodiff(im, niter, kappa, gamma, option)

	Arguments:
	        img       - input image
	        niterlist - if True, the image will be plotted on every iteration
	        kappa     - conduction coefficient 20-100 ?
	        gamma     - max value of .25 for stability
	        step      - tuple, the distance between adjacent pixels in (y,x)
	        option    - 1 Perona Malik diffusion equation No 1
	                    2 Perona Malik diffusion equation No 2

	Returns:
	        imgout   - diffused image.

	kappa controls conduction as a function of gradient.  If kappa is low
	small intensity gradients are able to block conduction and hence diffusion
	across step edges.  A large value reduces the influence of intensity
	gradients on conduction.

	gamma controls speed of diffusion (you usually want it at a maximum of
	0.25)

	step is used to scale the gradients in case the spacing between adjacent
	pixels differs in the x and y axes

	Diffusion equation 1 favours high contrast edges over low contrast ones.
	Diffusion equation 2 favours wide regions over smaller ones.

	Reference: 
	P. Perona and J. Malik. 
	Scale-space and edge detection using ansotropic diffusion.
	IEEE Transactions on Pattern Analysis and Machine Intelligence, 
	12(7):629-639, July 1990.

	Original MATLAB code by Peter Kovesi  
	School of Computer Science & Software Engineering
	The University of Western Australia
	pk @ csse uwa edu au
	<http://www.csse.uwa.edu.au>

	Translated to Python and optimised by Alistair Muldal
	Department of Pharmacology
	University of Oxford
	<alistair.muldal@pharm.ox.ac.uk>


        More dynamic python code by Arafat Hasan
        Department of Computer Science and Engineering
        Mawlana Bhashani Science and Technology
        Tanggail-1902, Bangladesh
        <opendoor.arafata[at]gmail[dot]com>
    
	June 2000  original version.       
	March 2002 corrected diffusion eqn No 2.
	July 2012 translated to Python
        January 2020, dynamic python

	"""

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    imgoutarr = {}
    last = 0
    niterlist.sort()
    niter = niterlist[len(niterlist) - 1]

    if len(niterlist) != 0 and niterlist[0] == 0:
        imgoutarr[0] = imgout.copy()
        last = last + 1

    for ii in np.arange(1, niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        if 0 < sigma:
            # deltaSf=flt.gaussian_filter(deltaS,sigma);
            # deltaEf=flt.gaussian_filter(deltaE,sigma);
            # deltaSf=flt.gaussian(deltaS,sigma);
            # deltaEf=flt.gaussian(deltaE,sigma);

            # opencv
            deltaSf = cv2.GaussianBlur(deltaS, (0, 0), sigma,
                                       cv2.BORDER_DEFAULT)
            deltaEf = cv2.GaussianBlur(deltaE, (0, 0), sigma,
                                       cv2.BORDER_DEFAULT)
        else:
            deltaSf = deltaS
            deltaEf = deltaE

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf / kappa)**2.) / step[0]
            gE = np.exp(-(deltaEf / kappa)**2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaSf / kappa)**2.) / step[0]
            gE = 1. / (1. + (deltaEf / kappa)**2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

        if last <= len(niterlist) and ii == niterlist[last]:
            imgoutarr[ii] = imgout.copy()
            last = last + 1

    imgoutarr[niter] = imgout
    return imgoutarr


def prewittOnDic(imgdic={}):
    prewittKernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittKernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewittX = {
        key: cv2.filter2D(value, -1, prewittKernelX)
        for key, value in imgdic.items()
    }
    prewittY = {
        key: cv2.filter2D(value, -1, prewittKernelY)
        for key, value in imgdic.items()
    }
    prewittlst = dict(prewittX.items() + prewittY.items() +
                      [(k, prewittX[k] + prewittY[k])
                       for k in set(prewittY) & set(prewittX)])
    return prewittlst


def sobelOnDic(imgdic={}):
    sobelX = {
        key: cv2.Sobel(value, cv2.CV_8U, 1, 0, ksize=3)
        for key, value in imgdic.items()
    }
    sobelY = {
        key: cv2.Sobel(value, cv2.CV_8U, 0, 1, ksize=3)
        for key, value in imgdic.items()
    }
    sobellst = dict(sobelX.items() + sobelY.items() +
                    [(k, sobelX[k] + sobelY[k])
                     for k in set(sobelY) & set(sobelX)])
    return sobellst


if __name__ == '__main__':
    path = "../../img/Valve_original_(1).png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img = img.astype('float32')
    # img=img[300:600,300:600]
    # m = np.mean(img)
    # s = np.std(img)
    # nimg = (img - m) / s

    nimg = img.copy()

    niterlist = [0, 20, 40, 60, 80, 100]
    ksizelist = [3, 5, 7, 9, 11]

    gaussianblurlst = {
        ksize: cv2.GaussianBlur(img, (ksize, ksize), cv2.BORDER_DEFAULT)
        for ksize in ksizelist
    }
    ksizelist.append(0)
    ksizelist.sort()
    gaussianblurlst[0] = img

    anisodifflst = anisodiff(img=nimg,
                             niterlist=niterlist,
                             kappa=30,
                             gamma=0.075,
                             step=(1, 1),
                             option=2)

    cannyGAUlst = {
        key: cv2.Canny(np.uint8(value), 100, 200)
        for key, value in gaussianblurlst.items()
    }
    prewittGAUlst = prewittOnDic(gaussianblurlst)
    sobelGAUlst = sobelOnDic(gaussianblurlst)

    cannyANIlst = {
        key: cv2.Canny(np.uint8(value), 100, 200)
        for key, value in anisodifflst.items()
    }
    prewittANIlst = prewittOnDic(anisodifflst)
    sobelANIlst = sobelOnDic(anisodifflst)

    fig = plt.figure("Compare Anisotropic Diffusion and Gaussian filter")

    axGau = fig.add_subplot(241)
    axGAUCanny = fig.add_subplot(242)
    axGAUPrewitt = fig.add_subplot(243)
    axGAUSobel = fig.add_subplot(244)

    axAniDiff = fig.add_subplot(245)
    axANICanny = fig.add_subplot(246)
    axANIPrewitt = fig.add_subplot(247)
    axANISobel = fig.add_subplot(248)
    plt.subplots_adjust(left=.05,
                        bottom=0,
                        right=.95,
                        top=1,
                        wspace=.13,
                        hspace=None)

    axGau.annotate("Gaussian",
                   xy=(0, 0.5),
                   xytext=(-axGau.yaxis.labelpad - 5, 0),
                   xycoords=axGau.yaxis.label,
                   textcoords='offset points',
                   size='large',
                   ha='right',
                   va='center',
                   rotation=90)
    axAniDiff.annotate("Anisotropic Diffusion",
                       xy=(0, 0.5),
                       xytext=(-axAniDiff.yaxis.labelpad - 5, 0),
                       xycoords=axAniDiff.yaxis.label,
                       textcoords='offset points',
                       size='large',
                       ha='right',
                       va='center',
                       rotation=90)

    ims = []

    for iternum in range(6):

        titleGau = plt.text(0.5,
                            1.01,
                            "Gaussian, ksize: " + str(ksizelist[iternum]) +
                            "*" + str(ksizelist[iternum]),
                            ha="center",
                            va="bottom",
                            color=[1, 0, 0],
                            transform=axGau.transAxes,
                            fontsize="large")

        titleGAUCanny = plt.text(0.5,
                                 1.01,
                                 "Canny, ksize: " + str(ksizelist[iternum]) +
                                 "*" + str(ksizelist[iternum]),
                                 ha="center",
                                 va="bottom",
                                 color=[1, 0, 0],
                                 transform=axGAUCanny.transAxes,
                                 fontsize="large")

        titleGAUPrewitt = plt.text(0.5,
                                   1.01,
                                   "Prewitt, ksize: " +
                                   str(ksizelist[iternum]) + "*" +
                                   str(ksizelist[iternum]),
                                   ha="center",
                                   va="bottom",
                                   color=[1, 0, 0],
                                   transform=axGAUPrewitt.transAxes,
                                   fontsize="large")

        titleGAUSobel = plt.text(0.5,
                                 1.01,
                                 "Sobel, ksize: " + str(ksizelist[iternum]) +
                                 "*" + str(ksizelist[iternum]),
                                 ha="center",
                                 va="bottom",
                                 color=[1, 0, 0],
                                 transform=axGAUSobel.transAxes,
                                 fontsize="large")

        titleAniDiff = plt.text(0.5,
                                1.01,
                                "AnisoDiff, iteration: " +
                                str(niterlist[iternum]),
                                ha="center",
                                va="bottom",
                                color=[1, 0, 0],
                                transform=axAniDiff.transAxes,
                                fontsize="large")

        titleANICanny = plt.text(0.5,
                                 1.01,
                                 "Canny, iteration: " + str(niterlist[iternum]),
                                 ha="center",
                                 va="bottom",
                                 color=[1, 0, 0],
                                 transform=axANICanny.transAxes,
                                 fontsize="large")

        titleANIPrewitt = plt.text(0.5,
                                   1.01,
                                   "Prewitt, iteration: " +
                                   str(niterlist[iternum]),
                                   ha="center",
                                   va="bottom",
                                   color=[1, 0, 0],
                                   transform=axANIPrewitt.transAxes,
                                   fontsize="large")

        titleANISobel = plt.text(0.5,
                                 1.01,
                                 "Sobel, iteration: " + str(niterlist[iternum]),
                                 ha="center",
                                 va="bottom",
                                 color=[1, 0, 0],
                                 transform=axANISobel.transAxes,
                                 fontsize="large")

        showGau = axGau.imshow(gaussianblurlst[ksizelist[iternum]], cmap='gray')

        showGAUCanny = axGAUCanny.imshow(cannyGAUlst[ksizelist[iternum]],
                                         cmap='gray')

        showGAUPrewitt = axGAUPrewitt.imshow(prewittGAUlst[ksizelist[iternum]],
                                             cmap='gray')

        showGAUSobel = axGAUSobel.imshow(sobelGAUlst[ksizelist[iternum]],
                                         cmap='gray')

        showAniDiff = axAniDiff.imshow(anisodifflst[niterlist[iternum]],
                                       cmap='gray')

        showANICanny = axANICanny.imshow(cannyANIlst[niterlist[iternum]],
                                         cmap='gray')

        showANIPrewitt = axANIPrewitt.imshow(prewittANIlst[niterlist[iternum]],
                                             cmap='gray')

        showANISobel = axANISobel.imshow(sobelANIlst[niterlist[iternum]],
                                         cmap='gray')

        ims.append([
            showGau, titleGau, showGAUCanny, titleGAUCanny, showGAUPrewitt,
            titleGAUPrewitt, showGAUSobel, titleGAUSobel, showAniDiff,
            titleAniDiff, showANICanny, titleANICanny, showANIPrewitt,
            titleANIPrewitt, showANISobel, titleANISobel
        ])

    ani = animation.ArtistAnimation(fig,
                                    ims,
                                    interval=2000,
                                    blit=False,
                                    repeat_delay=0)

    plt.show()
