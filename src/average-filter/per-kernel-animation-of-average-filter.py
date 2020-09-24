#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:37:12 2020

@author: arafat_hasan
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

numberOfTimesAverageFilterToApply = 40

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

ksizelist = [3, 5, 7, 9, 11, 13, 15, 17, 19]

fig = plt.figure("Per Kernel Animation of Average Filter",
                 figsize=(16, 9),
                 dpi=300,
                 facecolor='w',
                 edgecolor='k')

ax = fig.add_subplot(111)
ax.set_xticks([])
ax.set_yticks([])

plt.subplots_adjust(left=.05,
                    bottom=.1,
                    right=.95,
                    top=.90,
                    wspace=.13,
                    hspace=None)

ax.annotate("Average Filter",
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords='offset points',
            size='large',
            ha='right',
            va='center',
            rotation=90)


ims = []

im = plt.imshow(img, animated=True, cmap='gray', vmin=0, vmax=255)
title = plt.text(0.5,
                 1.05,
                 "No kernel applied",
                 ha="center",
                 va="bottom",
                 color=[.3, .3, .3],
                 transform=ax.transAxes,
                 fontsize="large")
ims.append([im, title])

for ksize in ksizelist:
    kernel = np.ones((ksize, ksize), np.float32)/(ksize * ksize)
    covoleOutput = cv2.filter2D(img, -1, kernel)
    im = plt.imshow(covoleOutput, animated=True, cmap='gray', vmin=0, vmax=255)
    title = plt.text(0.5,
                     1.05,
                     "Kernel size " + str(ksize) + "*" + str(ksize),
                     ha="center",
                     va="bottom",
                     color=[.3, .3, .3],
                     transform=ax.transAxes,
                     fontsize="large")
    ims.append([im, title])


ani = animation.ArtistAnimation(fig,
                                ims,
                                interval=1000,
                                blit=False,
                                repeat_delay=0)


"""
To save as mp4 format, this line have to be uncommented and 'ffmpeg' package
must be  installed in the system.
Video speed can be controlled by increasing or decreasing the fps.
"""

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=1, metadata=dict(artist='CSE MBSTU'), bitrate=1800)
# ani.save('per-kernel-animation-of-average-filter.mp4', writer=writer)
# print("[INFO] Animation has written in mp4 file successfully...")

plt.show()

print("[INFO] All operations finished successfully...")
