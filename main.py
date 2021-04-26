import lab4
from regularizedderivatives import regularized_values
from estimateT import estimate_T
from estimateD import est_D
from noisifier import noisifier, multiplicative_noisifier
from est_HL import est_HL
from update_L import update_L
from update_u import update_u
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
    SNR = 20
    m = 10

    # img = lab4.make_circle(128,128,64)
    img = lab4.get_cameraman().astype(float)

    # add noise
    noisy_img = noisifier(img,SNR)
    # noisy_img = multiplicative_noisifier(img, SNR)

    noisy_img = np.clip(noisy_img, 0, 255)
    L = noisy_img

    Jg, Jgdx, Jgdy = regularized_values(L,ksize=3,sigma=3)

    window_size = np.array([3, 3])
    T_img = estimate_T(Jgdx,Jgdy,window_size)

    D = est_D(T_img,m)
    for _ in range(epochs):
        HL = est_HL(L)
        L = update_L(L,delta_s, D, HL)

    fig,axs = plt.subplots(1, 3, constrained_layout=True)
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original image')
    axs[1].imshow(noisy_img,cmap='gray')
    axs[1].set_title('Noisy img, SNR = ' + str(SNR))
    axs[2].imshow(L,cmap='gray')
    axs[2].set_title('Improved noisy image')
    fig.suptitle(f"epochs = {epochs}, delta_s = {delta_s}, m = {m}", fontsize=16)

    plt.show()
    plt.close('all')

    """Inpainting with TV"""
    TV_epochs = 1500
    alpha = 0.95
    lam = 0.01
    visualize = True

    missing_info = np.random.randint(0, high=3, size=img.shape)
    missing_info = np.clip(missing_info,0,1)
    img_missing = np.multiply(img, missing_info)

    u = np.copy(img_missing)
    g = np.copy(img_missing)

    plt.figure(1)
    for i in range(1, TV_epochs+1):
        plt.clf()
        u = update_u(u, g, missing_info, alpha, lam, ksize=3, sigma=3)
        u = np.clip(u, 0, 255)

        if i % 30 == 0 and visualize:
            plt.imshow(u, cmap="gray")
            plt.title(f"iteration {i}")
            plt.pause(0.000001)

    figure1, axes1 = plt.subplots(2,2, constrained_layout=True)
    axes1[0,0].imshow(missing_info, cmap='gray')
    axes1[0,0].set_title('mask image')
    axes1[1,0].imshow(img, cmap='gray')
    axes1[1,0].set_title('Original image')
    axes1[0,1].imshow(img_missing, cmap='gray')
    axes1[0,1].set_title('image with missing info')
    axes1[1,1].imshow(u, cmap='gray')
    axes1[1,1].set_title('restored image')

    print(np.max(u))
    print(np.min(u))
    plt.show()


if __name__ == '__main__':
    main()