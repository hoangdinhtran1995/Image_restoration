import lab4
from regularizedderivatives import regularized_values
from estimateT import estimate_T
from estimateD import est_D
from noisifier import noisifier, multiplicative_noisifier
from est_HL import est_HL
from update_L import update_L
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    Params:
    delta_s = 0.1
    epochs = 10
    SNR = 20
    m = 1
    """
    delta_s = 0.1
    epochs = 10
    SNR = 70
    m = 10


    # img = lab4.make_circle(128,128,64)
    img = lab4.get_cameraman().astype(float)

    # add noise
    # noisy_img = noisifier(img,SNR)
    noisy_img = multiplicative_noisifier(img, SNR)

    noisy_img = np.clip(noisy_img, 0, 255)
    L = noisy_img

    Jg, Jgdx, Jgdy = regularized_values(L,ksize=3,sigma=3)

    window_size = np.array([3, 3])
    T_img = estimate_T(Jgdx,Jgdy,window_size)

    D = est_D(T_img,m)
    for _ in range(epochs):
        HL = est_HL(L)
        L = update_L(L,delta_s, D, HL)

    fig,axs= plt.subplots(1, 3, constrained_layout=True)
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original image')
    axs[1].imshow(noisy_img,cmap='gray')
    axs[1].set_title('Noisy img, SNR = ' + str(SNR))
    axs[2].imshow(L,cmap='gray')
    axs[2].set_title('Improved noisy image')
    fig.suptitle(f"epochs = {epochs}, delta_s = {delta_s}, m = {m}", fontsize=16)

    plt.show()


if __name__ == '__main__':
    main()