import numpy as np
from scipy.signal import convolve2d as conv2



def estimate_T(gdx, gdy, window_size):
    gdx_2 = np.square(gdx)
    gdxy = np.multiply(gdx,gdy)
    gdy_2 = np.square(gdy)
    window = np.ones(window_size)

    T = np.empty(np.shape(gdx) + (2, 2))

    T[...,0,0] = conv2(gdx_2, window, mode="same")
    T[...,0,1] = conv2(gdxy, window, mode="same")
    T[...,1,0] = T[...,0,1]
    T[...,1,1] = conv2(gdy_2, window, mode="same")
    return T

