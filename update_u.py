import numpy as np
from scipy.signal import convolve2d as conv2
from est_HL import est_HL

def update_u(u, g, chi, alpha, lam, ksize=3, sigma=3):
    lp = np.atleast_2d(np.exp(-0.5 * (np.arange(-ksize, ksize + 1) / sigma) ** 2))
    lp = lp / np.sum(lp)  # normalize the filter
    df = np.atleast_2d(-1.0 / np.square(sigma) * np.arange(-ksize, ksize + 1) * lp)  # 1d deriv filt

    u_x = conv2(conv2(u, df, mode="same"), lp.T, mode="same")
    u_y = conv2(conv2(u, lp, mode="same"), df.T, mode="same")

    grad_u = np.array([u_x, u_y])

    T = np.empty(np.shape(u) + (2, 2))
    T[...,0,0] = u_x**2
    T[...,0,1] = u_x*u_y
    T[...,1,0] = T[...,0,1]
    T[...,1,1] = u_y**2

    H_u = est_HL(u)

    a = (H_u[...,0,0] * T[...,1,1] - 2 * H_u[...,0,1] * T[...,0,1] + H_u[...,1,1] * T[...,0,0])
    b = np.linalg.norm(grad_u, axis=0) ** 3 + 1e-6

    u_out = u - alpha * (chi*(u-g)-lam * a / b)
    return u_out