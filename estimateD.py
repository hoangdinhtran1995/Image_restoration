import numpy as np

def est_D(T, m):
    lam, e = np.linalg.eig(T)
    alpha = np.exp(-lam / m)
    alpha_1 = alpha[:,:,0,None,None]
    alpha_2 = alpha[:,:,1,None,None]
    e_1 = e[:,:,0,None]
    e_2 = e[:,:,1,None]
    d_1 = alpha_1 * np.matmul(np.swapaxes(e_1, 3, 2), e_1)
    d_2 = alpha_2 * np.matmul(np.swapaxes(e_2, 3, 2), e_2)
    D = d_1 + d_2
    return D