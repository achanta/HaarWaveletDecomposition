'''
The function haar_decomposition takes a 2D color or grayscale images and
returns a 2D array containing multi-resolution haar wavelet coefficients.
The number of levels is a user-provided parameter.
'''


def haar_decomposition(image,levels=3):


    def one_level_decomp(img):

        H,W = img.shape[0],img.shape[1]
        hwt = np.zeros(img.shape,dtype=np.int64)

        Y,X = H//2,W//2

        ax = img[:,0:X*2:2,...]
        bx = img[:,1:X*2:2,...]

        # hwt[:,:X,...] = (ax+bx)
        hwt[:,:X,...] = np.right_shift(ax+bx,1)
        hwt[:,X:X*2,...] = ax-bx

        ay = hwt[0:Y*2:2,:,...]
        by = hwt[1:Y*2:2,:,...]

        # img[:Y,:,...] = (ay+by)
        img[:Y,:,...] = np.right_shift(ay+by,1)
        img[Y:Y*2,:,...] = ay-by

        return img

    def multi_scale_decomp(image,levels):
        img = image.copy()
        H,W = img.shape[0],img.shape[1]

        for s in range(levels):
            img[:(H>>s),:(W>>s),...] = one_level_decomp(img[:(H>>s),:(W>>s),...])

        return img

    return multi_scale_decomp(image,levels)
