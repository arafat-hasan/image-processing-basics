import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

fig = plt.figure()

ims = []

img1 = cv2.imread('../../img/circle-1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../../img/circle-2.png', cv2.IMREAD_GRAYSCALE)
img1 = img1.astype('float')
img2 = img2.astype('float')

im = plt.imshow(img1, animated=True, cmap='gray')
ims.append([im])
im = plt.imshow(img2, animated=True, cmap='gray')
ims.append([im])

im = plt.imshow(img1, animated=True, cmap='gray')
ims.append([im])
im = plt.imshow(img2, animated=True, cmap='gray')
ims.append([im])

im = plt.imshow(img1, animated=True, cmap='gray')
ims.append([im])
im = plt.imshow(img2, animated=True, cmap='gray')
ims.append([im])

im = plt.imshow(img1, animated=True, cmap='gray')
ims.append([im])
im = plt.imshow(img2, animated=True, cmap='gray')
ims.append([im])

ani = animation.ArtistAnimation(fig,
                                ims,
                                interval=5000,
                                blit=True,
                                repeat_delay=0)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
ani.save('anisotropic-diffusion-practice.mp4', writer=writer)
plt.show()
