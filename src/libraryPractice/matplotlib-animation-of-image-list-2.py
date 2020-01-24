import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

imgarr = []
imgarr.append(cv2.imread('../../img/circle-1.png', cv2.IMREAD_GRAYSCALE))
imgarr.append(cv2.imread('../../img/circle-2.png', cv2.IMREAD_GRAYSCALE))
imgarr.append(cv2.imread('../../img/circle-1.png', cv2.IMREAD_GRAYSCALE))
imgarr.append(cv2.imread('../../img/circle-2.png', cv2.IMREAD_GRAYSCALE))
titles = ["adsf", "qwer"]

fig = plt.figure("ArtistAnimation")
ax = fig.add_subplot(111)

ims = []

for iternum in range(2):
    title = plt.text(0.5,
                     1.01,
                     titles[iternum],
                     ha="center",
                     va="bottom",
                     color=np.random.rand(3),
                     transform=ax.transAxes,
                     fontsize="large")
    # text = ax.text(iternum, iternum, titles[iternum])
    scatter = ax.imshow(imgarr[iternum], cmap='gray')
    ims.append([
        # text,
        scatter,
        title,
    ])

ani = animation.ArtistAnimation(fig,
                                ims,
                                interval=250,
                                blit=False,
                                repeat_delay=0)
plt.show()
