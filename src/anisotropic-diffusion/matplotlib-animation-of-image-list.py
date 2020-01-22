import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

# fig = plt.figure()

ims = []

img1 = cv2.imread('../../img/circle-1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../../img/circle-2.png', cv2.IMREAD_GRAYSCALE)
img1 = img1.astype('float')
img2 = img2.astype('float')

plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('original')

plt.subplot(1, 2, 2)
im = plt.imshow(img1, animated=True, cmap='gray')
plt.title('1')
ims.append([im])
im = plt.imshow(img2, animated=True, cmap='gray')
ims.append([im])

plt.subplot(1, 2, 2)
im = plt.imshow(img1, animated=True, cmap='gray')
plt.title('2')
ims.append([im])
im = plt.imshow(img2, animated=True, cmap='gray')
ims.append([im])

plt.subplot(1, 2, 2)
im = plt.imshow(img1, animated=True, cmap='gray')
plt.title('3')
ims.append([im])
im = plt.imshow(img2, animated=True, cmap='gray')
ims.append([im])

plt.subplot(1, 2, 2)
im = plt.imshow(img1, animated=True, cmap='gray')
plt.title('4')
ims.append([im])
im = plt.imshow(img2, animated=True, cmap='gray')
ims.append([im])

ani = animation.ArtistAnimation(fig,
                                ims,
                                interval=500,
                                blit=True,
                                repeat_delay=0)

plt.show()
