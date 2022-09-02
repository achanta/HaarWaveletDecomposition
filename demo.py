'''
Demo showing forward and inverse Haar wavelet decomposition.
The input can be in color or grayscale.

Please note that in order to save as an image, a clipping operation
is performed on the Haar wavelets, which discards negative values.
'''

import numpy as np
from timeit import default_timer as timer
from PIL import Image
from haar_wavelets import haar_forward, haar_inverse



def main():

    ref = Image.open("./bee.png")
    ref = np.asarray(ref).astype(np.float32)

    LEVELS = 2
    start = timer()
    hwt = haar_forward(ref.copy(), LEVELS) # forward transform
    inv = haar_inverse(hwt.copy(), LEVELS)  # backward transform
    end = timer()

    Image.fromarray(hwt.clip(0,255).astype(np.uint8)).save("bee_wavelets_for.png")
    Image.fromarray(inv.clip(0,255).astype(np.uint8)).save("bee_wavelets_inv.png")
    print("Time taken in seconds for forward and inverse Haar wavelet transform:", end-start)

    
if __name__ == "__main__":
    main()
