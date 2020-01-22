import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def simulate(*args):
    global ground_state
    ground_state = np.random.randn(5,5)

simulate()

fig = plt.figure()
im = plt.imshow(ground_state, animated = True)

def update_fig(*args):
    global ground_state
    simulate(ground_state, 100000, 2.4)
    im.set_data(ground_state)
    return im

ani = animation.FuncAnimation(fig, update_fig, interval = 50)

plt.show()
