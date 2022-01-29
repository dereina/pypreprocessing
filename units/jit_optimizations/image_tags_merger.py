import numpy as np
import utils
from numba import jit

@jit(cache=True, nopython=False)
def fourierMerge(ponderation, fft_A, fft_B, img_max):
    if ponderation == 'direct':
        fft_merge = (fft_A * np.abs(fft_A)  + fft_B * np.abs(fft_B)) / (np.abs(fft_A) + np.abs(fft_B))

    elif ponderation == 'inverse':
        fft_merge = (fft_A * 1/np.abs(fft_A)  + fft_B * 1/np.abs(fft_B)) / (1/np.abs(fft_A) + 1/np.abs(fft_B))

    elif ponderation == 'sum':
        fft_merge = fft_A + fft_B 

    else:
        print("ponderation error")
        assert 0

    img_merge = np.abs(utils.inv_FFT_all_channel(fft_merge))
    img_merge= (img_max * img_merge/np.max(img_merge)).astype(np.uint8) #you can apply contrast funcition here if needed or provided a precomputed maximum first...
    return img_merge


@jit(cache=True, nopython=True)
def spatialMerge(ponderation, img3, img2):
    if ponderation == 'direct':
        return ((img3 * img3  + img2 * img2) / (img3 + img2)).astype(np.uint8)

    elif ponderation == 'inverse':
        return (2 / (1/img3 + 1/img2)).astype(np.uint8)

    elif ponderation == 'sum':
        return ((img3 + img2) / 2 ).astype(np.uint8)
    
    else:
        print("ponderation error")
        assert 0
    
    return None