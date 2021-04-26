import lab4
from regularizedderivatives import regularized_values
from estimateT import estimate_T
from estimateD import est_D
import matplotlib.pyplot as plt
import numpy as np

def main():
    img = lab4.get_cameraman()
    plt.figure(1)
    plt.imshow(img)

    Jg, Jgdx, Jgdy = regularized_values(img,ksize=3,sigma=3)
    plt.figure(2)
    plt.imshow(Jg)

    window_size = np.array([3, 3])
    T_img = estimate_T(Jgdx,Jgdy,window_size)
    plt.figure()
    plt.imshow(T_img[:,:,0,0])
    plt.figure()
    plt.imshow(T_img[:,:,0,1])
    plt.figure()
    plt.imshow(T_img[:,:,1,1])

    D = est_D(T_img,m=1)

    plt.figure()
    plt.imshow(D[:,:,0,0])
    plt.figure()
    plt.imshow(D[:,:,1,0])
    plt.figure()
    plt.imshow(D[:,:,1,1])

    plt.show()


if __name__ == '__main__':
    main()