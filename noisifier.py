import numpy as np

# adds gaussian noise to the img
def noisifier(img, snr):
    variance = np.var(img) / (10 ** (snr / 10))
    return img + np.random.normal(scale=variance**(1/2), size=np.shape(img))

def multiplicative_noisifier(img, snr):
    variance = np.var(img) / (10 ** (snr / 10))
    noise = np.random.normal(scale=variance**(1/2), size=np.shape(img))
    return np.power(10, np.log10(img) + noise)
