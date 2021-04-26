import numpy as np

def update_L(L, ds, D, HL):
    return L + ds / 2 * np.trace(np.matmul(D, HL), axis1=3, axis2=2)