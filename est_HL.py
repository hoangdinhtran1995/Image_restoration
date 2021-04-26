import numpy as np
from scipy.signal import convolve2d as conv2

def est_HL(L):
    h11 = np.array([
        [0, 0, 0],
        [1, -2, 1],
        [0, 0, 0]])
    h12 = np.array([
        [1/2, 0, -1/2],
        [0, 0, 0],
        [-1/2, 0, 1/2]])
    h22 = h11.T

    HL = np.empty(np.shape(L) + (2, 2))
    HL[:,:,0,0] = conv2(L, h11, mode="same")
    HL[:,:,0,1] = conv2(L, h12, mode="same")
    HL[:,:,1,0] = HL[...,0,1]
    HL[:,:,1,1] = conv2(L, h22, mode="same")
    return HL